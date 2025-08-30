#!/usr/bin/env python3

import json
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from IPython.display import Markdown, display


class BedrockChatbot:

    def __init__(self):
        self.session = None

    def send_prompt(self, prompt_data, temperature=0.0, top_p=1.0, max_token_count=1000):
        # Inference
        bedrock_inference = self.session.client(service_name='bedrock-runtime')

        # Claude 3.5 Sonnet v2 request format
        body = json.dumps(
            {
                "inputText": prompt_data,
                "textGenerationConfig": {
                    "temperature": temperature,
                    "topP": top_p,
                    "maxTokenCount": max_token_count,
                },
            }
        )

        # Claude 3.5 Sonnet v2 model ID
        model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
        accept = "application/json"
        contentType = "application/json"

        # Try Claude models in order of preference, starting with 3.5 Sonnet v1 (more stable)
        model_ids_to_try = [
            model_id
        ]
        
        for model_id in model_ids_to_try:
            try:
                print(f"🔄 Trying model: {model_id}")
                response = bedrock_inference.invoke_model(
                    body=body,
                    modelId=model_id,
                    accept=accept,
                    contentType=contentType
                )

                response_body = json.loads(response['body'].read())
                # Claude returns content in a different format
                if 'content' in response_body and len(response_body['content']) > 0:
                    print(f"✅ Successfully used: {model_id}")
                    return response_body['content'][0]['text']
                else:
                    print(f"❓ No content in response from {model_id}")
                    continue
                    
            except ClientError as e:
                error_code = e.response['Error']['Code']
                error_message = e.response['Error']['Message']
                print(f"❌ Failed with {model_id}: {error_code}: {error_message}")
                
                # If it's just an access issue, try the next model
                if error_code in ['AccessDeniedException', 'ValidationException']:
                    continue

            except Exception as e:
                print(f"❌ Unexpected error with {model_id}: {str(e)}")
                continue


    def connect_to_bedrock(self):
        try:
            # Specify region explicitly - Bedrock is available in specific regions
            region_name = 'us-west-2'  # Change this to your preferred region

            print(f"Attempting to connect to Bedrock in region: {region_name}")

            # Create session and client
            self.session = boto3.Session()
            bedrock = self.session.client(
                service_name='bedrock',
                region_name=region_name
            )

            # Test the connection
            print("Testing connection...")
            models = bedrock.list_foundation_models()["modelSummaries"]
            print(f"Available models: {len(models)}")
            
            # Show first few models
            for model in models[:3]:
                print(f"  - {model['modelName']} ({model['modelId']})")

            print("✅ Successfully connected to Bedrock!")
            return True

        except NoCredentialsError:
            print("❌ Error: AWS credentials not found.")
            print("Please configure your credentials using:")
            print("  aws configure")
            return False

        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']

            if error_code == 'UnrecognizedClientException':
                print("❌ Error: Invalid AWS credentials or expired token")
                print("Solutions:")
                print("  1. Run 'aws configure' to set up new credentials")
                print("  2. If using temporary credentials, refresh them")
                print("  3. Check if your AWS account has Bedrock access")

            elif error_code == 'AccessDeniedException':
                print("❌ Error: Access denied to Bedrock")
                print("Your AWS account may not have Bedrock enabled or proper permissions")

            else:
                print(f"❌ AWS Error ({error_code}): {error_message}")
            return False

        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            return False


def check_aws_config():
    """Check current AWS configuration"""
    try:
        session = boto3.Session()
        print(f"Current AWS Region: {session.region_name}")
        print(f"Available profiles: {session.available_profiles}")
        
        # Try to get caller identity
        sts = session.client('sts')
        identity = sts.get_caller_identity()
        print(f"AWS Account: {identity.get('Account')}")
        print(f"User ARN: {identity.get('Arn')}")
        
    except Exception as e:
        print(f"Cannot verify AWS configuration: {e}")


if __name__ == '__main__':
    chatbot = BedrockChatbot()
    
    print("Connecting to Bedrock...")
    if chatbot.connect_to_bedrock():
        prompt_data = (
            "Can you name a few real-life applications of natural language processing?"
        )

        print(f"\n📝 Prompt: {prompt_data}")
        print("\n🤖 Response:")
        
        try:
            response = chatbot.send_prompt(prompt_data, temperature=0.0)
            print(response)
        except Exception as e:
            print(f"❌ Error generating response: {e}")
    else:
        print("❌ Failed to connect to Bedrock. Please check your credentials and try again.")
