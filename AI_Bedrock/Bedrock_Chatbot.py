#!/usr/bin/env python3

import json
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from IPython.display import Markdown, display


class BedrockChatbot:

    def __init__(self):
        pass

    def send_prompt(self, prompt_data, temperature = 0.0, top_p=1.0, max_token_count=1000):
        # Inference
        bedrock_inference = self.session.client(service_name='bedrock-runtime')

        body = json.dumps(
            {
                'inputTex': prompt_data,
                'textGenerationConfig': {
                    'temperature': temperature,
                    'topP': top_p,
                    'maxTokenCount': max_token_count
                }

            }
        )

        model_id = 'amazon.titan-text-express-v1'
        accept = "application/json"
        contentType = "application/json"

        response = bedrock_inference.invoke_model(body = body,
                                                  model_id = model_id, accept=accept,
                                                  contentType= contentType)

        response_body = json.loads(response['body'].read())
        return response_body['results'][0]['outputText']

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
            print(bedrock.list_foundation_models()["modelSummaries"][0:3])

            print("✅ Successfully connected to Bedrock!")

        except NoCredentialsError:
            print("❌ Error: AWS credentials not found.")
            print("Please configure your credentials using:")
            print("  aws configure")

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

        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")

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
    # print("=== AWS Bedrock Connection Test ===")
    # print()
    
    # print("1. Checking AWS Configuration...")
    # # check_aws_config()
    # print()

    chatbot = BedrockChatbot()
    
    print("Connecting to Bedrock...")
    chatbot.connect_to_bedrock()

    prompt_data = (
        """Can you name a few real-life applications of natural language processing?"""
    )

    display(Markdown(prompt_data))
    display(Markdown(chatbot.send_prompt(prompt_data, temperature=0.0)))
