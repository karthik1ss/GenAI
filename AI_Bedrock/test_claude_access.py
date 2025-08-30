#!/usr/bin/env python3

import json
import boto3
from botocore.exceptions import ClientError

def test_claude_models():
    try:
        session = boto3.Session()
        bedrock = session.client(service_name='bedrock', region_name='us-west-2')
        bedrock_runtime = session.client(service_name='bedrock-runtime', region_name='us-west-2')
        
        # Get all Claude models
        response = bedrock.list_foundation_models()
        claude_models = [m for m in response['modelSummaries'] 
                        if 'claude' in m['modelId'].lower() and 'TEXT' in m['outputModalities']]
        
        print(f"Found {len(claude_models)} Claude text models:")
        for model in claude_models:
            print(f"  - {model['modelName']} ({model['modelId']}) - Status: {model['modelLifecycle']['status']}")
        
        print("\n🔍 Testing Claude 3.5 Sonnet v2 access...")
        
        # Test Claude 3.5 Sonnet v2 models specifically
        claude_35_v2_models = [m for m in claude_models 
                              if 'claude-3-5-sonnet-20241022-v2' in m['modelId']]
        
        if not claude_35_v2_models:
            print("❌ No Claude 3.5 Sonnet v2 models found")
            return
        
        print(f"Found {len(claude_35_v2_models)} Claude 3.5 Sonnet v2 models:")
        for model in claude_35_v2_models:
            print(f"  - {model['modelId']}")
        
        # Test each Claude 3.5 Sonnet v2 model
        for model in claude_35_v2_models:
            model_id = model['modelId']
            print(f"\n🧪 Testing {model_id}...")
            
            # Claude request format
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, can you respond with just 'Hello back!'?"
                    }
                ],
                "temperature": 0.0
            })
            
            try:
                response = bedrock_runtime.invoke_model(
                    body=body,
                    modelId=model_id,
                    accept='application/json',
                    contentType='application/json'
                )
                
                response_body = json.loads(response['body'].read())
                if 'content' in response_body and len(response_body['content']) > 0:
                    text_response = response_body['content'][0]['text']
                    print(f"  ✅ SUCCESS: {text_response}")
                    return model_id  # Return the first working model
                else:
                    print(f"  ❓ Unexpected response format: {response_body}")
                    
            except ClientError as e:
                error_code = e.response['Error']['Code']
                error_message = e.response['Error']['Message']
                print(f"  ❌ {error_code}: {error_message}")
            except Exception as e:
                print(f"  ❌ Unexpected error: {str(e)}")
        
        print("\n❌ No Claude 3.5 Sonnet v2 models are accessible")
        return None
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == '__main__':
    print("🤖 Testing Claude 3.5 Sonnet v2 Access")
    print("=" * 50)
    working_model = test_claude_models()
    
    if working_model:
        print(f"\n✅ Working model found: {working_model}")
    else:
        print("\n❌ No working Claude models found")
