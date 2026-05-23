"""First example of Claude tool use.

This is a minimal example to understand the tool use protocol BEFORE
building the full agent. Demonstrates:
1. How to define a tool with JSON schema
2. How Claude returns a tool_use block
3. How to execute the tool and feed results back
4. How Claude generates a final response

Run with: python -m scripts.hello_tool_use
"""
from anthropic import Anthropic

from src.config import settings
from src.logger import logger

# 1. Define our single tool
TOOLS = [
  {
      "name": "get_aws_service_info",
      "description": (
          "Returns basic information about an AWS service. "
          "Use this when the user asks about a specific AWS service."
      ),
      "input_schema": {
          "type": "object",
          "properties": {
              "service_name": {
                  "type": "string",
                  "description": "The AWS service name (e.g., 'Lambda', 'S3', 'DynamoDB')",
              }
          },
          "required": ["service_name"],
      },
  }
]


# 2. Implement the actual tool function (just a mock for now)
def get_aws_service_info(service_name: str) -> str:
  """Mock implementation. Real version would query AWS docs/RAG."""
  info_db = {
      "Lambda": "AWS Lambda is a serverless compute service that runs code in response to events.",
      "S3": "Amazon S3 is an object storage service with industry-leading scalability and durability.",
      "DynamoDB": "DynamoDB is a fully managed NoSQL database service.",
      "Bedrock": "Amazon Bedrock is a fully managed service for foundation models.",
  }
  return info_db.get(
      service_name,
      f"No information available for '{service_name}'.",
  )


# 3. Tool execution dispatcher
def execute_tool(tool_name: str, tool_input: dict) -> str:
  """Map tool name to function and execute it."""
  if tool_name == "get_aws_service_info":
      return get_aws_service_info(**tool_input)
  return f"Unknown tool: {tool_name}"


def main():
  client = Anthropic(api_key=settings.anthropic_api_key)

  user_question = "What is AWS Lambda and what are its main use cases?"

  print(f"\n🤔 User: {user_question}\n")

  # Initial messages list
  messages = [{"role": "user", "content": user_question}]

  # Agent loop (max 5 iterations for this hello world)
  for iteration in range(5):
      logger.info("agent_iteration", iteration=iteration + 1)

      # Call Claude with our tools available
      response = client.messages.create(
          model=settings.claude_model,
          max_tokens=settings.max_tokens,
          tools=TOOLS,
          messages=messages,
      )

      logger.info(
          "claude_response",
          stop_reason=response.stop_reason,
          input_tokens=response.usage.input_tokens,
          output_tokens=response.usage.output_tokens,
      )

      # If Claude finished without using tools, we're done
      if response.stop_reason == "end_turn":
          # Extract the text response
          final_text = ""
          for block in response.content:
              if block.type == "text":
                  final_text += block.text
          print(f"\n✅ Claude (final): {final_text}\n")
          return

      # If Claude wants to use tools, execute them
      if response.stop_reason == "tool_use":
          # Add Claude's response to messages
          messages.append({"role": "assistant", "content": response.content})

          # Process each tool_use block
          tool_results = []
          for block in response.content:
              if block.type == "tool_use":
                  logger.info(
                      "tool_use_requested",
                      tool_name=block.name,
                      tool_input=block.input,
                  )

                  # Execute the tool
                  result = execute_tool(block.name, block.input)

                  logger.info(
                      "tool_executed",
                      tool_name=block.name,
                      result_preview=result[:100],
                  )

                  tool_results.append({
                      "type": "tool_result",
                      "tool_use_id": block.id,
                      "content": result,
                  })

          # Feed tool results back to Claude
          messages.append({"role": "user", "content": tool_results})

  print("\n⚠️  Max iterations reached without final answer.\n")


if __name__ == "__main__":
  main()
