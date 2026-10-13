# Winnow-12B server with the wrapper baked in, for pushing to ECR (scripts/ecr_push.sh). BASE is the image compose
# builds from EldanRing/winnow-inference; nothing from this repo is mounted at run time.
ARG BASE=winnow-inference:a50c7d0
FROM ${BASE}
COPY serve.py guard.py /wrap/
ENTRYPOINT ["python3", "/wrap/serve.py", "--model", "q8"]
