from api_calls import query_llm_deep_infra
from user_select import user_input

llm = user_input()

response = query_llm_deep_infra(
    retrieved_text="poop",
    user_query="poop",
    llm=llm
)

print(response)
