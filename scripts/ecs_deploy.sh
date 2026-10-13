#!/usr/bin/env bash
# Run the service 24/7 on ECS: one GPU instance in an Auto Scaling group, one always-running task (Winnow-12B + API).
#
#   scripts/ecs_deploy.sh            # first time: roles, cluster, GPU capacity, task definition, service
#   IMAGE_TAG=<new> scripts/ecs_deploy.sh   # later: registers a new task definition and rolls the service onto it
#
# Needs the images in ECR (scripts/ecr_push.sh), the model file in S3, and AWS credentials that can manage IAM, EC2,
# Auto Scaling and ECS. Settings come from the environment or .env:
#   AWS_REGION, ECR_REGISTRY, IMAGE_TAG     from scripts/ecr_push.sh
#   ECR_PREFIX        names everything (cluster, roles, repos, parameter path); default voice2decisions
#   MODEL_S3_URI      s3://bucket/path/Winnow-12B-Q8_0.gguf; copied to the instance's disk on first boot
#   SUBNET_IDS        comma-separated subnets for the instance (public ones if the UI calls it over the internet)
#   ALLOWED_CIDR      who may reach the API on port 8000 (default: nowhere until you set it, e.g. your UI's range)
#   SECURITY_GROUP_ID use this instead of creating one; INSTANCE_TYPE (default g5.2xlarge)
# The task definition is ecs/task-definition.json; the instance boot script is ecs/user-data.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
step() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
command -v aws >/dev/null || die "aws CLI not found"

export AWS_REGION=${AWS_REGION:?set AWS_REGION} PREFIX=${ECR_PREFIX:-voice2decisions}
export ECR_REGISTRY=${ECR_REGISTRY:?set ECR_REGISTRY (scripts/ecr_push.sh prints it)} IMAGE_TAG=${IMAGE_TAG:?set IMAGE_TAG}
export MODEL_S3_URI=${MODEL_S3_URI:?set MODEL_S3_URI, e.g. s3://my-bucket/models/Winnow-12B-Q8_0.gguf}
export MODEL_S3_OBJECT=${MODEL_S3_URI#s3://} MODEL_SHA256=b710efc4c0d048ee61eed92c5fef5ce323a4d17e7c51f9f0533cc72ae50818ea
export CLUSTER=$PREFIX
subnets=${SUBNET_IDS:?set SUBNET_IDS (comma-separated)}
instance_type=${INSTANCE_TYPE:-g5.2xlarge}
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
work=$(mktemp -d); trap 'rm -rf "$work"' EXIT
aws() { command aws --region "$AWS_REGION" "$@"; }
render() {  # ecs/<file> with ${NAME} filled from the exported settings above (and SECRETS)
  python3 - "$1" <<'PY'
import os, string, sys
keys = "PREFIX AWS_REGION ACCOUNT ECR_REGISTRY IMAGE_TAG EXECUTION_ROLE_ARN SECRETS CLUSTER MODEL_S3_URI MODEL_S3_OBJECT MODEL_SHA256"
print(string.Template(open(sys.argv[1]).read()).safe_substitute({k: os.environ.get(k, "") for k in keys.split()}))
PY
}
role() {  # name, service that assumes it
  if ! aws iam get-role --role-name "$1" >/dev/null 2>&1; then
    aws iam create-role --role-name "$1" --assume-role-policy-document \
      "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Principal\":{\"Service\":\"$2\"},\"Action\":\"sts:AssumeRole\"}]}" >/dev/null
    echo "created role $1"
  fi
}

step "IAM roles"
role "$PREFIX-ecs-execution" ecs-tasks.amazonaws.com
aws iam attach-role-policy --role-name "$PREFIX-ecs-execution" --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
aws iam put-role-policy --role-name "$PREFIX-ecs-execution" --policy-name secrets --policy-document "$(render ecs/execution-role-policy.json)"
export EXECUTION_ROLE_ARN="arn:aws:iam::$ACCOUNT:role/$PREFIX-ecs-execution"
role "$PREFIX-ecs-instance" ec2.amazonaws.com
aws iam attach-role-policy --role-name "$PREFIX-ecs-instance" --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role
aws iam attach-role-policy --role-name "$PREFIX-ecs-instance" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
aws iam put-role-policy --role-name "$PREFIX-ecs-instance" --policy-name model-file --policy-document "$(render ecs/instance-role-policy.json)"
if ! aws iam get-instance-profile --instance-profile-name "$PREFIX-ecs-instance" >/dev/null 2>&1; then
  aws iam create-instance-profile --instance-profile-name "$PREFIX-ecs-instance" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$PREFIX-ecs-instance" --role-name "$PREFIX-ecs-instance"
  echo "created instance profile; waiting for IAM to catch up"; sleep 15
fi

step "Let both roles pull the images"
for repo in api winnow; do  # adds them to the repository policy scripts/ecr_push.sh set, if it set one
  current=$(aws ecr get-repository-policy --repository-name "$PREFIX/$repo" --query policyText --output text 2>/dev/null || true)
  [ -n "$current" ] || { echo "$PREFIX/$repo: no repository policy, nothing to add"; continue; }
  merged=$(printf '%s' "$current" | python3 -c '
import json, sys
p, add = json.load(sys.stdin), sys.argv[1:]
for st in p["Statement"]:
    if st.get("Sid") == "TeamPull":
        st["Principal"]["AWS"] = sorted(set([st["Principal"]["AWS"]] if isinstance(st["Principal"]["AWS"], str) else st["Principal"]["AWS"]) | set(add))
    if st.get("Sid") == "OnlyTheTeam":
        arns = st["Condition"]["ArnNotEquals"]["aws:PrincipalArn"]
        st["Condition"]["ArnNotEquals"]["aws:PrincipalArn"] = sorted(set([arns] if isinstance(arns, str) else arns) | set(add))
print(json.dumps(p))' "$EXECUTION_ROLE_ARN" "arn:aws:iam::$ACCOUNT:role/$PREFIX-ecs-instance")
  for try in 1 2 3 4 5 6; do  # a role created a moment ago may not be accepted as a principal yet
    aws ecr set-repository-policy --repository-name "$PREFIX/$repo" --policy-text "$merged" >/dev/null 2>&1 && break
    [ "$try" = 6 ] && die "couldn't update the $PREFIX/$repo repository policy"
    sleep 10
  done
  echo "$PREFIX/$repo: execution and instance roles may pull"
done

step "Cluster $CLUSTER and logs"
aws ecs create-cluster --cluster-name "$CLUSTER" >/dev/null
aws logs create-log-group --log-group-name "/ecs/$PREFIX" 2>/dev/null || true

step "Security group"
sg=${SECURITY_GROUP_ID:-}
if [ -z "$sg" ]; then
  vpc=$(aws ec2 describe-subnets --subnet-ids "${subnets%%,*}" --query 'Subnets[0].VpcId' --output text)
  sg=$(aws ec2 describe-security-groups --filters Name=group-name,Values="$PREFIX-ecs" Name=vpc-id,Values="$vpc" \
         --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
  if [ -z "$sg" ] || [ "$sg" = None ]; then
    sg=$(aws ec2 create-security-group --group-name "$PREFIX-ecs" --vpc-id "$vpc" \
           --description "voice-2-decisions API" --query GroupId --output text)
    echo "created $sg"
  fi
  if [ -n "${ALLOWED_CIDR:-}" ]; then
    aws ec2 authorize-security-group-ingress --group-id "$sg" --protocol tcp --port 8000 --cidr "$ALLOWED_CIDR" 2>/dev/null || true
    echo "port 8000 open to $ALLOWED_CIDR"
  else
    echo "port 8000 is closed: set ALLOWED_CIDR (your UI's addresses) and run again"
  fi
fi

step "GPU capacity: $instance_type, ECS GPU-optimized AMI"
ami=$(aws ssm get-parameter --name /aws/service/ecs/optimized-ami/amazon-linux-2023/gpu/recommended/image_id \
        --query Parameter.Value --output text 2>/dev/null \
      || aws ssm get-parameter --name /aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended/image_id \
        --query Parameter.Value --output text)
echo "AMI $ami"
render ecs/user-data.sh | base64 -w0 > "$work/user-data.b64"
python3 - "$ami" "$instance_type" "$PREFIX" "$sg" "$work/user-data.b64" > "$work/lt.json" <<'PY'
import json, sys
ami, itype, prefix, sg, ud = sys.argv[1:]
print(json.dumps({"ImageId": ami, "InstanceType": itype, "IamInstanceProfile": {"Name": f"{prefix}-ecs-instance"},
                  "SecurityGroupIds": [sg], "UserData": open(ud).read(),
                  "MetadataOptions": {"HttpTokens": "required", "HttpPutResponseHopLimit": 2},
                  "BlockDeviceMappings": [{"DeviceName": "/dev/xvda", "Ebs": {"VolumeSize": 100, "VolumeType": "gp3",
                                                                              "Encrypted": True}}],
                  "TagSpecifications": [{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": f"{prefix}-gpu"}]}]}))
PY
if aws ec2 describe-launch-templates --launch-template-names "$PREFIX-gpu" >/dev/null 2>&1; then
  aws ec2 create-launch-template-version --launch-template-name "$PREFIX-gpu" --launch-template-data "file://$work/lt.json" >/dev/null
  aws ec2 modify-launch-template --launch-template-name "$PREFIX-gpu" --default-version '$Latest' >/dev/null
  echo "launch template updated (new instances use it; the running one keeps its settings)"
else
  aws ec2 create-launch-template --launch-template-name "$PREFIX-gpu" --launch-template-data "file://$work/lt.json" >/dev/null
  echo "launch template created"
fi
if ! aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names "$PREFIX-gpu" \
       --query 'AutoScalingGroups[0].AutoScalingGroupName' --output text | grep -q "$PREFIX-gpu"; then
  aws autoscaling create-auto-scaling-group --auto-scaling-group-name "$PREFIX-gpu" \
    --launch-template "LaunchTemplateName=$PREFIX-gpu,Version=\$Default" \
    --min-size 1 --max-size 1 --desired-capacity 1 --vpc-zone-identifier "$subnets" \
    --tags "Key=AmazonECSManaged,Value=true,PropagateAtLaunch=true"
  echo "Auto Scaling group created (1 instance)"
fi
asg_arn=$(aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names "$PREFIX-gpu" \
            --query 'AutoScalingGroups[0].AutoScalingGroupARN' --output text)
if ! aws ecs describe-capacity-providers --capacity-providers "$PREFIX-gpu" \
       --query 'capacityProviders[0].status' --output text 2>/dev/null | grep -q ACTIVE; then
  aws ecs create-capacity-provider --name "$PREFIX-gpu" --auto-scaling-group-provider \
    "autoScalingGroupArn=$asg_arn,managedScaling={status=ENABLED,targetCapacity=100,minimumScalingStepSize=1,maximumScalingStepSize=1},managedTerminationProtection=DISABLED" >/dev/null
  aws ecs put-cluster-capacity-providers --cluster "$CLUSTER" --capacity-providers "$PREFIX-gpu" \
    --default-capacity-provider-strategy "capacityProvider=$PREFIX-gpu,weight=1" >/dev/null
  echo "capacity provider $PREFIX-gpu attached to $CLUSTER"
fi

step "Task definition (secrets: every Parameter Store parameter under /$PREFIX/)"
export SECRETS=$(aws ssm get-parameters-by-path --path "/$PREFIX/" --recursive --query 'Parameters[].[Name,ARN]' --output json \
  | python3 -c "import json,sys; print(json.dumps([{'name': n.rsplit('/',1)[-1], 'valueFrom': a} for n, a in json.load(sys.stdin)]))")
echo "$SECRETS" | python3 -c "import json,sys; print('  ' + (', '.join(s['name'] for s in json.load(sys.stdin)) or 'none'))"
render ecs/task-definition.json > "$work/task.json"
taskdef=$(aws ecs register-task-definition --cli-input-json "file://$work/task.json" \
            --query taskDefinition.taskDefinitionArn --output text)
echo "$taskdef"

step "Service $PREFIX (desired count 1)"
if aws ecs describe-services --cluster "$CLUSTER" --services "$PREFIX" --query 'services[0].status' --output text 2>/dev/null | grep -q ACTIVE; then
  aws ecs update-service --cluster "$CLUSTER" --service "$PREFIX" --task-definition "$taskdef" --desired-count 1 >/dev/null
  echo "rolling onto $taskdef"
else
  # One GPU instance holds one task, so a deploy stops the old task before starting the new one (a few minutes down).
  aws ecs create-service --cluster "$CLUSTER" --service-name "$PREFIX" --task-definition "$taskdef" --desired-count 1 \
    --capacity-provider-strategy "capacityProvider=$PREFIX-gpu,weight=1" \
    --deployment-configuration "minimumHealthyPercent=0,maximumPercent=100" >/dev/null
  echo "created"
fi

cat <<MSG

Started. When you next run scripts/ecr_push.sh, keep these in PULL_ROLE_ARNS so ECS can still pull:
  $EXECUTION_ROLE_ARN,arn:aws:iam::$ACCOUNT:role/$PREFIX-ecs-instance
The first start waits for the instance to boot and copy the model from S3. Watch it with:
  aws ecs wait services-stable --region $AWS_REGION --cluster $CLUSTER --services $PREFIX
  aws logs tail /ecs/$PREFIX --region $AWS_REGION --follow
Then the API is at http://<instance address>:8000 (EC2 console, instance "$PREFIX-gpu").
MSG
