"""
Example demonstrating the various Message types available in Cortex.

Message types are used to structure conversations between users, AI, and tools.
They provide type safety, metadata tracking, and proper formatting.
"""

from cortex import (
    Message,
    SystemMessage,
    DeveloperMessage,
    UserMessage,
    AIMessage,
    ToolMessage,
    ToolMessageGroup,
    MessageUsage,
    AgentUsage,
    FunctionCall,
    InputImage,
    InputFile,
)


# Example 1: Basic Message Types
def basic_message_types():
    """Demonstrate the core message types."""
    
    # Base Message (rarely used directly)
    msg = Message(content="This is a base message")
    print(f"Message: {msg.content}")
    
    # SystemMessage - for system prompts
    system_msg = SystemMessage(content="You are a helpful assistant.")
    print(f"System: {system_msg.content}")
    
    # DeveloperMessage - for developer instructions
    dev_msg = DeveloperMessage(content="Use formal language in your response.")
    print(f"Developer: {dev_msg.content}")
    
    # UserMessage - for user input
    user_msg = UserMessage(content="Hello, how are you?")
    print(f"User: {user_msg.build_content()}")
    
    # AIMessage - for AI responses
    ai_msg = AIMessage(
        content="I'm doing well, thank you!",
        model="gpt-4o-mini"
    )
    print(f"AI ({ai_msg.model}): {ai_msg.content}")
    
    # ToolMessage - for tool results
    tool_msg = ToolMessage(
        content="Result: 42",
        tool_call_id="call_123"
    )
    print(f"Tool: {tool_msg.content}")


# Example 2: UserMessage with Images and Files
def user_message_with_attachments():
    """Demonstrate UserMessage with images and files."""
    
    # Create image inputs
    image1 = InputImage(
        image_url="https://example.com/image1.jpg",
        detail="high"
    )
    
    image2 = InputImage(
        file_id="file-abc123",
        detail="auto"
    )
    
    # Create file inputs
    file1 = InputFile(
        filename="document.pdf",
        file_url="https://example.com/doc.pdf"
    )
    
    file2 = InputFile(
        filename="data.json",
        file_data='{"key": "value"}'
    )
    
    # UserMessage with attachments
    user_msg = UserMessage(
        content="Please analyze these images and files",
        images=[image1, image2],
        files=[file1, file2]
    )
    
    print(f"User message: {user_msg.content}")
    print(f"Images attached: {len(user_msg.images)}")
    print(f"Files attached: {len(user_msg.files)}")


# Example 3: AIMessage with Function Calls
def ai_message_with_function_calls():
    """Demonstrate AIMessage with function calls."""
    
    # Create function calls
    func_call1 = FunctionCall(
        id="call_1",
        type="function",
        call_id="call_1",
        name="get_weather",
        arguments='{"location": "New York"}'
    )
    
    func_call2 = FunctionCall(
        id="call_2",
        type="function",
        call_id="call_2",
        name="search_web",
        arguments='{"query": "Python tutorials"}'
    )
    
    # AIMessage with function calls
    ai_msg = AIMessage(
        content="I'll check the weather and search for tutorials.",
        model="gpt-4o",
        function_calls=[func_call1, func_call2]
    )
    
    print(f"AI: {ai_msg.content}")
    print(f"Function calls: {len(ai_msg.function_calls)}")
    for fc in ai_msg.function_calls:
        print(f"  - {fc.name}({fc.arguments})")


# Example 4: ToolMessageGroup
def tool_message_group_example():
    """Demonstrate grouping multiple tool results."""
    
    # Create individual tool messages
    tool_msg1 = ToolMessage(
        content="Weather in New York: 72°F, Sunny",
        tool_call_id="call_1"
    )
    
    tool_msg2 = ToolMessage(
        content="Found 10 Python tutorials",
        tool_call_id="call_2"
    )
    
    tool_msg3 = ToolMessage(
        content="Current time: 14:30 UTC",
        tool_call_id="call_3"
    )
    
    # Group them together
    tool_group = ToolMessageGroup(
        tool_messages=[tool_msg1, tool_msg2, tool_msg3]
    )
    
    print("Tool Results Group:")
    for i, tm in enumerate(tool_group.tool_messages, 1):
        print(f"  {i}. {tm.content}")


# Example 5: MessageUsage Tracking
def message_usage_tracking():
    """Demonstrate token usage tracking."""
    
    # Create usage for a single message
    usage1 = MessageUsage(
        prompt_tokens=100,
        completion_tokens=50,
        cached_tokens=20,
        total_tokens=150
    )
    
    print("Message 1 Usage:")
    print(usage1.format())
    
    # Create another usage
    usage2 = MessageUsage(
        prompt_tokens=80,
        completion_tokens=40,
        total_tokens=120
    )
    
    # Accumulate usage
    usage1.accumulate(usage2)
    
    print("Accumulated Usage:")
    print(usage1.format())


# Example 6: AgentUsage Tracking
def agent_usage_tracking():
    """Demonstrate tracking usage across multiple models."""
    
    # Create agent usage tracker
    agent_usage = AgentUsage()
    
    # Add usage for different models
    gpt4_usage = MessageUsage(
        prompt_tokens=500,
        completion_tokens=200,
        total_tokens=700
    )
    agent_usage.add_usage("gpt-4o", gpt4_usage)
    
    gpt4_mini_usage = MessageUsage(
        prompt_tokens=300,
        completion_tokens=100,
        total_tokens=400
    )
    agent_usage.add_usage("gpt-4o-mini", gpt4_mini_usage)
    
    # Add more usage to the same model
    more_gpt4_usage = MessageUsage(
        prompt_tokens=200,
        completion_tokens=80,
        total_tokens=280
    )
    agent_usage.add_usage("gpt-4o", more_gpt4_usage)
    
    print("Agent Usage Summary:")
    print(agent_usage.format())


# Example 7: Using Messages in Conversations
def conversation_example():
    """Demonstrate building a conversation with different message types."""
    
    conversation = []
    
    # System prompt
    conversation.append(
        SystemMessage(content="You are a helpful coding assistant.")
    )
    
    # Developer instructions
    conversation.append(
        DeveloperMessage(content="Provide code examples when relevant.")
    )
    
    # User question
    conversation.append(
        UserMessage(
            content="How do I read a file in Python?"
        )
    )
    
    # AI response with function call
    conversation.append(
        AIMessage(
            content="I'll search for the best practices.",
            model="gpt-4o-mini",
            function_calls=[
                FunctionCall(
                    id="call_1",
                    type="function",
                    call_id="call_1",
                    name="search_docs",
                    arguments='{"topic": "file reading python"}'
                )
            ]
        )
    )
    
    # Tool result
    conversation.append(
        ToolMessage(
            content="Found: Use open() with context manager",
            tool_call_id="call_1"
        )
    )
    
    # Final AI response
    conversation.append(
        AIMessage(
            content="Here's how to read a file:\n```python\nwith open('file.txt', 'r') as f:\n    content = f.read()\n```",
            model="gpt-4o-mini"
        )
    )
    
    print("Conversation Flow:")
    for i, msg in enumerate(conversation, 1):
        msg_type = type(msg).__name__
        content_preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
        print(f"{i}. {msg_type}: {content_preview}")


# Example 8: Message Decoration
def message_decoration_example():
    """Demonstrate message decoration for display."""
    
    messages = [
        SystemMessage(content="System prompt"),
        DeveloperMessage(content="Developer note"),
        UserMessage(content="User question"),
        AIMessage(content="AI response"),
        ToolMessage(content="Tool result", tool_call_id="call_1"),
    ]
    
    print("Decorated Messages (for rich console output):")
    for msg in messages:
        print(msg.decorate())


if __name__ == "__main__":
    print("=" * 80)
    print("Example 1: Basic Message Types")
    print("=" * 80)
    basic_message_types()
    
    print("\n" + "=" * 80)
    print("Example 2: UserMessage with Attachments")
    print("=" * 80)
    user_message_with_attachments()
    
    print("\n" + "=" * 80)
    print("Example 3: AIMessage with Function Calls")
    print("=" * 80)
    ai_message_with_function_calls()
    
    print("\n" + "=" * 80)
    print("Example 4: ToolMessageGroup")
    print("=" * 80)
    tool_message_group_example()
    
    print("\n" + "=" * 80)
    print("Example 5: MessageUsage Tracking")
    print("=" * 80)
    message_usage_tracking()
    
    print("\n" + "=" * 80)
    print("Example 6: AgentUsage Tracking")
    print("=" * 80)
    agent_usage_tracking()
    
    print("\n" + "=" * 80)
    print("Example 7: Conversation Example")
    print("=" * 80)
    conversation_example()
    
    print("\n" + "=" * 80)
    print("Example 8: Message Decoration")
    print("=" * 80)
    message_decoration_example()
