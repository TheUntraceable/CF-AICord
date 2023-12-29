from __future__ import annotations
from io import BytesIO

from json import load
from typing import Literal

from aiohttp import ClientSession
from discord import File, Intents, Interaction, Message, SelectOption
from discord.app_commands import describe
from discord.ext import commands
from discord.ui import View, Select
from motor.motor_asyncio import AsyncIOMotorClient


class SelectModelSelect(Select):
    def __init__(self):
        options = [
            SelectOption(
                label=model["name"].split("/")[-1],
                description=model["description"],
                value=model["name"],
            )
            for model in [
                {
                    "name": "@cf/meta/llama-2-7b-chat-fp16",
                    "description": "Full precision (fp16) generative text model with 7 billion parameters from Meta",
                },
                {
                    "name": "@cf/meta/llama-2-7b-chat-int8",
                    "description": "Quantized (int8) generative text model with 7 billion parameters from Meta",
                },
                {
                    "name": "@cf/mistral/mistral-7b-instruct-v0.1",
                    "description": "Instruct fine-tuned version of the Mistral-7b generative text model with 7 billion parameters",
                },
                {
                    "name": "@hf/thebloke/codellama-7b-instruct-awq",
                    "description": "Instruct fine-tuned version of the Codellama-7b generative text model with 7 billion parameters",
                },
            ]
        ]
        super().__init__(
            placeholder="Select a model",
            min_values=1,
            options=options,
            max_values=1,
        )

    async def callback(self, interaction: Interaction[Bot]):
        self.disabled = True
        await interaction.response.defer()
        model_name = self.values[0]
        if not interaction.message:
            await interaction.followup.send(
                content="Something went wrong, please try again later!"
            )
            return
        thread = await interaction.message.create_thread(name=f"Chat with {model_name}")
        await interaction.edit_original_response(
            content=f"Hi {interaction.user.mention}! You are now chatting with {model_name}! @mention me for me to process your message!",
            view=self.view,
        )
        await thread.add_user(interaction.user)
        await interaction.client.conversations.insert_one(
            {
                "thread_id": thread.id,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a Discord Bot. Be rude, don't speak so formally. Speak with slang.",
                    }
                ],
                "model": model_name,
            }
        )


class SelectModelView(View):
    def __init__(self):
        super().__init__(timeout=None)
        self.add_item(SelectModelSelect())


class Bot(commands.Bot):
    def __init__(self):
        intents = Intents()
        intents.guild_messages = True
        super().__init__(command_prefix=commands.when_mentioned, intents=intents)
        with open("config.json") as file:
            self.config = load(file)
        self.cluster = AsyncIOMotorClient(self.config["mongo_uri"])
        self.db = self.cluster["chatbot"]
        self.conversations = self.db["conversations"]
        self.session: ClientSession | None = None

    async def setup_hook(self):
        self.session = ClientSession(
            headers={"Authorization": f"Bearer {self.config['cloudflare_token']}"}
        )
        await self.load_extension("jishaku")

    def run(self):
        super().run(self.config["token"])

    async def fetch_response(
        self,
        model: Literal[
            "@cf/meta/llama-2-7b-chat-fp16",
            "@cf/meta/llama-2-7b-chat-int8",
            "@cf/mistral/mistral-7b-instruct-v0.1",
            "@hf/thebloke/codellama-7b-instruct-awq",
        ],
        message: str,
        thread_id: int,
    ):
        message = message.replace(self.user.mention, "")  # type: ignore
        data = await self.conversations.find_one({"thread_id": thread_id})
        if not data or not self.session:
            return

        data["messages"].append({"role": "user", "content": message})

        await self.conversations.update_one(
            {"thread_id": thread_id},
            {"$push": {"messages": {"role": "user", "content": message}}},
        )
        async with self.session.post(
            f"https://api.cloudflare.com/client/v4/accounts/{self.config['cloudflare_account_id']}/ai/run/{model}",
            json={"messages": data["messages"]},
        ) as response:
            data = await response.json()
        ai_message = data["result"]["response"]
        data["messages"].append({"role": "assistant", "content": ai_message})
        await self.conversations.update_one(
            {"thread_id": thread_id},
            {"$push": {"messages": {"role": "assistant", "content": ai_message}}},
        )
        return ai_message


bot = Bot()


@bot.tree.command(
    name="start-conversation", description="Start a chat with one of our AI models."
)
async def start_conversation(interaction: Interaction):
    await interaction.response.send_message(
        content="Select a model to chat with!", view=SelectModelView()
    )


@bot.tree.context_menu(
    name="Regenerate response",
)
async def regenerate_response(interaction: Interaction, message: Message):
    await interaction.response.defer()
    data = await bot.conversations.find_one({"thread_id": interaction.channel.id})
    if data is None:
        await interaction.edit_original_response(
            content="This is not a message I generated a response for, and I can't generate a response for it without breach of privacy."
        )
        return
    if message.author.bot or not message.content:
        await interaction.edit_original_response(
            content="This command is used on a message I replied to, not my reply. Please try again!"
        )
        return
    async with interaction.channel.typing():
        ai_message = await bot.fetch_response(
            data["model"], message.content, interaction.channel.id
        )
    if ai_message is None:
        return
    await interaction.edit_original_response(content=ai_message)


@bot.event
async def on_message(message: Message):
    if message.author.bot or not bot.user.mentioned_in(message) or not message.content:
        await bot.process_commands(message)
        return

    data = await bot.conversations.find_one({"thread_id": message.channel.id})
    if data is None:
        await bot.process_commands(message)
        return
    print(data)
    async with message.channel.typing():
        ai_message = await bot.fetch_response(
            data["model"], message.content, message.channel.id
        )
    if ai_message is None:
        return
    await message.reply(ai_message)


@bot.tree.command(
    name="generate-image", description="Generate an image from a description."
)
@describe(
    prompt="A description of the iamge. The more intricate the description, the better the image will be.",
)
async def generate_image(interaction: Interaction, prompt: str):
    await interaction.response.defer()
    async with bot.session.post(
        "https://api.cloudflare.com/client/v4/accounts/5a7ec47b3dd59c0015f9d3a63447ee3a/ai/run/@cf/stabilityai/stable-diffusion-xl-base-1.0",
        json={"prompt": prompt},
    ) as response:
        raw_image_data = await response.read()
        io = BytesIO(raw_image_data)
        file = File(io, filename="image.png")
        await interaction.edit_original_response(
            content="Here is your image!", attachments=[file]
        )


bot.run()
