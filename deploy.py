"""
deploy.py — one-command deployment of the RAGAS demo to AWS EC2.

Usage:
    python deploy.py

What it does:
  1. Zips the application code
  2. Creates an S3 bucket and uploads the zip
  3. Creates an IAM role + instance profile with Bedrock permissions
  4. Creates a security group (ports 8501 and 22)
  5. Launches an EC2 t3.medium (Amazon Linux 2023) with a user-data script
     that installs Python 3.11, downloads the zip from S3, and starts Streamlit
  6. Waits for the instance to be running and prints the public URL
"""

import io
import json
import os
import time
import zipfile

import boto3
from botocore.exceptions import ClientError

# ── config ────────────────────────────────────────────────────────────────────
PROFILE       = "brijesh"
REGION        = "us-east-1"
APP_NAME      = "ragas-demo"
INSTANCE_TYPE = "t3.medium"
KEY_NAME      = f"{APP_NAME}-key"        # set to your existing key pair name or leave as-is to create one
APP_PORT      = 8501
SSH_PORT      = 22

# Files/dirs to include in the zip (chroma_db is excluded — re-ingested on EC2)
INCLUDE_EXTENSIONS = {".py", ".txt", ".jsonl", ".md"}
EXCLUDE_DIRS       = {"chroma_db", "__pycache__", ".git", "deploy"}

# ── helpers ───────────────────────────────────────────────────────────────────
def _session():
    return boto3.Session(profile_name=PROFILE, region_name=REGION)


def _account_id(sts):
    return sts.get_caller_identity()["Account"]


# ── step 1: zip the app ───────────────────────────────────────────────────────
def create_zip() -> bytes:
    print("[1/6] Zipping application code…")
    buf = io.BytesIO()
    base = os.path.dirname(os.path.abspath(__file__))
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(base):
            # prune excluded dirs in-place
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for fname in files:
                _, ext = os.path.splitext(fname)
                if ext in INCLUDE_EXTENSIONS:
                    full = os.path.join(root, fname)
                    arcname = os.path.relpath(full, base)
                    zf.write(full, arcname)
                    print(f"    + {arcname}")
    size_kb = buf.tell() // 1024
    print(f"  Zip size: {size_kb} KB")
    return buf.getvalue()


# ── step 2: S3 upload ─────────────────────────────────────────────────────────
def upload_to_s3(s3, account_id: str, zip_bytes: bytes) -> str:
    bucket = f"{APP_NAME}-{account_id}"
    key    = "ragas.zip"
    print(f"[2/6] Uploading to s3://{bucket}/{key}…")

    # create bucket if needed
    try:
        if REGION == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(Bucket=bucket,
                             CreateBucketConfiguration={"LocationConstraint": REGION})
        print(f"  Created bucket: {bucket}")
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            raise
        print(f"  Bucket already exists: {bucket}")

    s3.put_object(Bucket=bucket, Key=key, Body=zip_bytes)
    print("  Upload complete.")
    return bucket


# ── step 3: IAM role ──────────────────────────────────────────────────────────
def ensure_iam_role(iam) -> str:
    role_name    = f"{APP_NAME}-ec2-role"
    profile_name = f"{APP_NAME}-ec2-profile"
    print(f"[3/6] Ensuring IAM role '{role_name}'…")

    trust = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    })

    try:
        iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=trust,
                        Description="RAGAS demo EC2 role")
        print(f"  Created role: {role_name}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "EntityAlreadyExists":
            raise
        print(f"  Role already exists: {role_name}")

    for policy_arn in [
        "arn:aws:iam::aws:policy/AmazonBedrockFullAccess",
        "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess",
    ]:
        try:
            iam.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
            print(f"  Attached: {policy_arn.split('/')[-1]}")
        except ClientError:
            pass

    try:
        iam.create_instance_profile(InstanceProfileName=profile_name)
        iam.add_role_to_instance_profile(InstanceProfileName=profile_name, RoleName=role_name)
        print(f"  Created instance profile: {profile_name}")
        time.sleep(10)  # IAM propagation delay
    except ClientError as e:
        if e.response["Error"]["Code"] != "EntityAlreadyExists":
            raise
        print(f"  Instance profile already exists: {profile_name}")

    return profile_name


# ── step 4: security group ────────────────────────────────────────────────────
def ensure_security_group(ec2, vpc_id: str) -> str:
    sg_name = f"{APP_NAME}-sg"
    print(f"[4/6] Ensuring security group '{sg_name}'…")

    existing = ec2.describe_security_groups(
        Filters=[{"Name": "group-name", "Values": [sg_name]}])["SecurityGroups"]
    if existing:
        sg_id = existing[0]["GroupId"]
        print(f"  Using existing SG: {sg_id}")
        return sg_id

    sg = ec2.create_security_group(
        GroupName=sg_name,
        Description="RAGAS demo - Streamlit port 8501 and SSH",
        VpcId=vpc_id,
    )
    sg_id = sg["GroupId"]
    ec2.authorize_security_group_ingress(GroupId=sg_id, IpPermissions=[
        {"IpProtocol": "tcp", "FromPort": APP_PORT, "ToPort": APP_PORT,
         "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "Streamlit"}]},
        {"IpProtocol": "tcp", "FromPort": SSH_PORT, "ToPort": SSH_PORT,
         "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "SSH"}]},
    ])
    print(f"  Created SG: {sg_id} (ports {APP_PORT}, {SSH_PORT})")
    return sg_id


# ── step 5: get latest Amazon Linux 2023 AMI ─────────────────────────────────
def get_al2023_ami(ec2) -> str:
    print("[5/6] Finding latest Amazon Linux 2023 AMI…")
    resp = ec2.describe_images(
        Owners=["amazon"],
        Filters=[
            {"Name": "name",                "Values": ["al2023-ami-2023.*-x86_64"]},
            {"Name": "state",               "Values": ["available"]},
            {"Name": "architecture",        "Values": ["x86_64"]},
            {"Name": "virtualization-type", "Values": ["hvm"]},
        ],
    )
    images = sorted(resp["Images"], key=lambda i: i["CreationDate"], reverse=True)
    ami_id = images[0]["ImageId"]
    print(f"  AMI: {ami_id}  ({images[0]['Name']})")
    return ami_id


# ── step 6: launch EC2 ────────────────────────────────────────────────────────
def ensure_key_pair(ec2) -> str:
    """Use existing key pair or create a new one, saving the .pem file locally."""
    existing = ec2.describe_key_pairs().get("KeyPairs", [])
    if existing:
        kp = existing[0]["KeyName"]
        print(f"  Using existing key pair: {kp}")
        return kp

    print(f"  No key pairs found. Creating '{KEY_NAME}'…")
    resp = ec2.create_key_pair(KeyName=KEY_NAME)
    pem_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{KEY_NAME}.pem")
    with open(pem_path, "w") as f:
        f.write(resp["KeyMaterial"])
    os.chmod(pem_path, 0o400)
    print(f"  Key saved to: {pem_path}  (keep this safe!)")
    return KEY_NAME


def launch_ec2(ec2, ami_id: str, sg_id: str, profile_name: str, bucket: str, key_name: str) -> dict:
    print(f"[6/6] Launching EC2 {INSTANCE_TYPE}…")

    user_data = f"""#!/bin/bash
set -e
exec > /var/log/ragas-setup.log 2>&1
echo "=== RAGAS setup started at $(date) ==="

# Update and install Python 3.11
dnf update -y
dnf install -y python3.11 python3.11-pip unzip

# Download app from S3
mkdir -p /home/ec2-user/ragas
cd /home/ec2-user/ragas
aws s3 cp s3://{bucket}/ragas.zip . --region {REGION}
unzip -o ragas.zip

# Install Python dependencies (Python 3.11 avoids 3.14 wheel issues)
python3.11 -m pip install --upgrade pip
python3.11 -m pip install -r requirements.txt --prefer-binary

# Unset AWS_PROFILE so EC2 uses IAM role instead
unset AWS_PROFILE
export AWS_DEFAULT_REGION={REGION}

# Start Streamlit
nohup python3.11 -m streamlit run app.py \\
    --server.port {APP_PORT} \\
    --server.address 0.0.0.0 \\
    --server.headless true \\
    --browser.gatherUsageStats false \\
    > /var/log/streamlit.log 2>&1 &

echo "=== RAGAS setup complete at $(date) ==="
"""

    kwargs = dict(
        ImageId=ami_id,
        InstanceType=INSTANCE_TYPE,
        MinCount=1, MaxCount=1,
        SecurityGroupIds=[sg_id],
        IamInstanceProfile={"Name": profile_name},
        UserData=user_data,
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [{"Key": "Name", "Value": APP_NAME}],
        }],
    )
    if key_name:
        kwargs["KeyName"] = key_name

    resp     = ec2.run_instances(**kwargs)
    instance = resp["Instances"][0]
    iid      = instance["InstanceId"]
    print(f"  Instance launched: {iid}")
    print("  Waiting for running state…")

    waiter = ec2.get_waiter("instance_running")
    waiter.wait(InstanceIds=[iid])

    info       = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]
    public_ip  = info.get("PublicIpAddress", "N/A")
    public_dns = info.get("PublicDnsName", "N/A")
    return {"instance_id": iid, "public_ip": public_ip, "public_dns": public_dns, "key_name": key_name}


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  RAGAS DEMO — AWS DEPLOYMENT")
    print(f"  Profile : {PROFILE}")
    print(f"  Region  : {REGION}")
    print(f"  Instance: {INSTANCE_TYPE}")
    print("=" * 60)

    sess       = _session()
    ec2        = sess.client("ec2")
    s3         = sess.client("s3")
    iam        = sess.client("iam")
    sts        = sess.client("sts")

    account_id = _account_id(sts)
    print(f"AWS Account: {account_id}\n")

    zip_bytes  = create_zip()
    bucket     = upload_to_s3(s3, account_id, zip_bytes)
    profile    = ensure_iam_role(iam)

    # Get default VPC
    vpcs   = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"]
    vpc_id = vpcs[0]["VpcId"]

    sg_id    = ensure_security_group(ec2, vpc_id)
    key_name = ensure_key_pair(ec2)
    ami_id   = get_al2023_ami(ec2)
    info     = launch_ec2(ec2, ami_id, sg_id, profile, bucket, key_name)

    print("\n" + "=" * 60)
    print("  DEPLOYMENT COMPLETE")
    print("=" * 60)
    print(f"  Instance ID : {info['instance_id']}")
    print(f"  Public IP   : {info['public_ip']}")
    print(f"\n  App URL: http://{info['public_ip']}:{APP_PORT}")
    print(f"\n  Note: Allow ~3 minutes for setup to finish on the instance.")
    print(f"  SSH : ssh -i {info['key_name']}.pem ec2-user@{info['public_ip']}")
    print(f"  Logs: ssh in and run: tail -f /var/log/ragas-setup.log")
    print("=" * 60)


if __name__ == "__main__":
    main()
