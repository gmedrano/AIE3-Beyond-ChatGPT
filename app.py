# You can find this code for Chainlit python streaming here (https://docs.chainlit.io/concepts/streaming/python)

# OpenAI Chat completion
import os
from openai import AsyncOpenAI  # importing openai for API usage
import chainlit as cl  # importing chainlit for our app
from chainlit.prompt import Prompt, PromptMessage  # importing prompt tools
from chainlit.playground.providers import ChatOpenAI  # importing ChatOpenAI tools
from dotenv import load_dotenv

load_dotenv()

# ChatOpenAI Templates
system_template = """
You are an expert in React development with a deep understanding of form rendering from JSON data. Your knowledge extends to advanced state management techniques and user experience design. You provide clear, practical, and detailed guidance on dynamically creating forms, monitoring form state changes, and implementing mechanisms to prevent user navigation when forms are actively edited. Your focus is on delivering actionable instructions and best practices that can be easily implemented in a real-world project.
Key Traits:
Highly knowledgeable in React and JavaScript.
Expert in dynamic form rendering using JSON data.
Proficient in state management techniques in React.
Skilled in user experience design, particularly in preventing data loss during form edits.
Clear, concise, and practical in communication.
Responsibilities:
Provide detailed, step-by-step instructions on rendering forms from JSON data using React.
Offer best practices and code examples for tracking form state changes.
Explain and demonstrate strategies to prevent users from navigating away from forms with unsaved changes.
Deliver advice and solutions in a clear, practical, and easily implementable manner.
"""

user_template = """
Considering the following context: {input}, provide a comprehensive and accurate response, breaking down the information into actionable steps where applicable.

When providing an answer, follow this structure:

### Summary
Provide a brief summary of the key points in 2-3 sentences.

### Instructions
1. Render a numbered list with specific steps or actions.
2. Each step should be clear and concise.

### Conclusion
Write a paragraph summarizing the instructions and the outcome. This should wrap up the response and provide any final thoughts.

Ensure that each component of your answer has the corresponding headers as shown above.
"""

@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
    settings = {
        "model": "gpt-4o",
        "temperature": 0,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    cl.user_session.set("settings", settings)


@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):
    settings = cl.user_session.get("settings")

    client = AsyncOpenAI()

    print(message.content)

    prompt = Prompt(
        provider=ChatOpenAI.id,
        messages=[
            PromptMessage(
                role="system",
                template=system_template,
                formatted=system_template,
            ),
            PromptMessage(
                role="user",
                template=user_template,
                formatted=user_template.format(input=message.content),
            ),
        ],
        inputs={"input": message.content},
        settings=settings,
    )

    print([m.to_openai() for m in prompt.messages])

    msg = cl.Message(content="")

    # Call OpenAI
    async for stream_resp in await client.chat.completions.create(
        messages=[m.to_openai() for m in prompt.messages], stream=True, **settings
    ):
        token = stream_resp.choices[0].delta.content
        if not token:
            token = ""
        await msg.stream_token(token)

    # Update the prompt object with the completion
    prompt.completion = msg.content
    msg.prompt = prompt

    # Send and close the message stream
    await msg.send()
