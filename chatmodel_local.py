from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

# (optional but helps speed)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Load small fast model
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.2,
        max_new_tokens=50   # keep small for fast response
    )
)

# Wrap as chat model
model = ChatHuggingFace(llm=llm)

# Ask question
result = model.invoke("galgotia university  is located in which city  ?")
print(result.content)