from openai import OpenAI
from aws_agent import get_monthly_bill, find_idle_ec2_instances

client = OpenAI()

def ask_ai(prompt):
    # Basic question understanding
    if "bill" in prompt.lower():
        data = get_monthly_bill()
    elif "idle" in prompt.lower():
        data = find_idle_ec2_instances()
    else:
        data = "Sorry, I can only answer about bills or idle resources right now."

    # Ask GPT to generate a conversational explanation
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI cloud assistant helping with AWS insights."},
            {"role": "user", "content": f"Explain this data: {data}"}
        ]
    )

    return response.choices[0].message.content
