# Model licenses

The model file is not stored in this repository; you copy it onto the instance (see README).

| Model | Used for | License | Source |
|---|---|---|---|
| Winnow-12B Q8_0 (EldanRing, fine-tune of Google Gemma 4 12B IT) | decisions: what the caller wants, whether a tail number / customer name was said, fuzzy-match tie-breaks | Apache-2.0 | https://huggingface.co/EldanRing/Winnow-12B |
| GPT 6 Luna (OpenAI, on Amazon Bedrock), optional | pulling tail numbers and customer names out of a line | AWS / OpenAI service terms | https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-6-luna.html |

Runtime, from GitHub:

| Runtime | License | Source |
|---|---|---|
| winnow-inference @ a50c7d0 (built from source; includes llama.cpp) | MIT | https://github.com/EldanRing/winnow-inference |

Winnow-12B's license allows use and redistribution, provided its license text and notices travel with any copy you
make (for example in an S3 bucket used with `MODELS_SOURCE=url`).
