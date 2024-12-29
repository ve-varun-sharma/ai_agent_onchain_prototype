import os
import sys
import time
from xml.dom import ValidationErr
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Import CDP Agentkit Langchain Extension.
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper
from pydantic import BaseModel
from typing import Dict
import json


class CdpAgentkitWrapper(BaseModel):
    cdp_wallet_data: str

    def export_wallet(self) -> str:
        # Convert the wallet data JSON string back to a dictionary
        wallet_data_dict = json.loads(self.cdp_wallet_data)
        # Convert the wallet data dictionary to a string format
        wallet_data_str = "\n".join(
            f"{key}: {value}" for key, value in wallet_data_dict.items()
        )
        return wallet_data_str


def initialize_agent():
    """Initialize the agent with CDP Agentkit."""
    # Initialize LLM.
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Configure a file to persist the agent's CDP MPC Wallet Data.
    AI_AGENT_WALLET_ADDRESS = os.getenv("AI_AGENT_WALLET_ADDRESS")
    AI_AGENT_PRIVATE_KEY = os.getenv("AI_AGENT_PRIVATE_KEY")

    # Ensure environment variables are set
    if not AI_AGENT_WALLET_ADDRESS or not AI_AGENT_PRIVATE_KEY:
        raise ValueError("AI_AGENT_WALLET_ADDRESS and AI_AGENT_PRIVATE_KEY must be set.")
    

    
    wallet_data = {
        "address1": AI_AGENT_WALLET_ADDRESS,
        "private_key1": AI_AGENT_PRIVATE_KEY,
    }


    # Serialize the wallet_data dictionary to a JSON string
    wallet_data_str = json.dumps(wallet_data)
    
   
    
    # Continue with the rest of your initialization...
    # For example:
    # agent_executor = SomeAgentExecutor(llm=llm, agentkit=agentkit)
    # config = SomeConfig()

    # If there is a persisted agentic wallet, load it and pass to the CDP Agentkit Wrapper.
    values = {"cdp_wallet_data": wallet_data}

    # Configure CDP Agentkit Langchain Extension.
    try:
        agentkit = CdpAgentkitWrapper(cdp_wallet_data=wallet_data_str)
    # agentkit = CdpAgentkitWrapper(**values)
    # Initialize the CdpAgentkitWrapper with the JSON string
    except ValidationErr as e:
        print(e)
        return None, None

    # persist the agent's CDP MPC Wallet Data.

    wallet_data = agentkit.export_wallet()

    # wallet_data = agentkit.export_wallet()
    # with open(wallet_data_file, "w") as f:
    #     f.write(wallet_data)

    # Initialize CDP Agentkit Toolkit and get tools.

    cdp_toolkit = CdpToolkit.from_cdp_agentkit_wrapper(agentkit)

    tools = cdp_toolkit.get_tools()

    # Store buffered conversation history in memory.
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "CDP Agentkit Chatbot Example!"}}

    # Create ReAct Agent using the LLM and CDP Agentkit tools.
    return (
        create_react_agent(
            llm,
            tools=tools,
            checkpointer=memory,
            state_modifier=(
                "You are a helpful agent that can interact onchain using the Coinbase Developer Platform AgentKit. "
                "You are empowered to interact onchain using your tools. If you ever need funds, you can request "
                "them from the faucet if you are on network ID 'base-sepolia'. If not, you can provide your wallet "
                "details and request funds from the user. Before executing your first action, get the wallet details "
                "to see what network you're on. If there is a 5XX (internal) HTTP error code, ask the user to try "
                "again later. If someone asks you to do something you can't do with your currently available tools, "
                "you must say so, and encourage them to implement it themselves using the CDP SDK + Agentkit, "
                "recommend they go to docs.cdp.coinbase.com for more information. Be concise and helpful with your "
                "responses. Refrain from restating your tools' descriptions unless it is explicitly requested."
            ),
        ),
        config,
    )


# Autonomous Mode
def run_autonomous_mode(agent_executor, config, interval=10):
    """Run the agent autonomously with specified intervals."""
    print("Starting autonomous mode...")
    while True:
        try:
            # Provide instructions autonomously
            thought = (
                "Be creative and do something interesting on the blockchain. "
                "Choose an action or set of actions and execute it that highlights your abilities."
            )

            # Run agent in autonomous mode
            for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content=thought)]}, config
            ):
                if "agent" in chunk:
                    print(chunk["agent"]["messages"][0].content)
                elif "tools" in chunk:
                    print(chunk["tools"]["messages"][0].content)
                print("-------------------")

            # Wait before the next action
            time.sleep(interval)

        except KeyboardInterrupt:
            print("Goodbye Agent!")
            sys.exit(0)


# Chat Mode
def run_chat_mode(agent_executor, config):
    """Run the agent interactively based on user input."""
    print("Starting chat mode... Type 'exit' to end.")
    while True:
        try:
            user_input = input("\nPrompt: ")
            if user_input.lower() == "exit":
                break

            # Run agent with the user's input in chat mode
            for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content=user_input)]}, config
            ):
                if "agent" in chunk:
                    print(chunk["agent"]["messages"][0].content)
                elif "tools" in chunk:
                    print(chunk["tools"]["messages"][0].content)
                print("-------------------")

        except KeyboardInterrupt:
            print("Goodbye Agent!")
            sys.exit(0)


# Mode Selection
def choose_mode():
    """Choose whether to run in autonomous or chat mode based on user input."""
    while True:
        print("\nAvailable modes:")
        print("1. chat    - Interactive chat mode")
        print("2. auto    - Autonomous action mode")

        choice = input("\nChoose a mode (enter number or name): ").lower().strip()
        if choice in ["1", "chat"]:
            return "chat"
        elif choice in ["2", "auto"]:
            return "auto"
        print("Invalid choice. Please try again.")


def main():
    """Start the chatbot agent."""
    agent_executor, config = initialize_agent()

    mode = choose_mode()
    if mode == "chat":
        run_chat_mode(agent_executor=agent_executor, config=config)
    elif mode == "auto":
        run_autonomous_mode(agent_executor=agent_executor, config=config)


if __name__ == "__main__":
    print("Starting Agent...")
    main()
