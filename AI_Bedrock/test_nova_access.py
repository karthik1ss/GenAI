#!/usr/bin/env python3

import json
import boto3
from botocore.exceptions import ClientError

def test_nova_models():
    try:
        session = boto3.Session()
        bedrock = session.client(service_name='bedrock', region_name='us-west-2')
        bedrock_runtime = session.client(service_name='bedrock-runtime', region_name='us-west-2')
        
        # Get Nova models
        response = bedrock.list_foundation_models()
        nova_models = [m for m in response['modelSummaries'] 
                      if 'nova' in m['modelId'].lower() and 'TEXT' in m['outputModalities']]
        
        print(f"Found {len(nova_models)} Nova text models:")
        for model in nova_models:
            print(f"  - {model['modelName']} ({model['modelId']}) - Status: {model['modelLifecycle']['status']}")
        
        print("\n🔍 Testing Nova model access...")
        
        # Test Nova models
        test_models = [
            'amazon.nova-micro-v1:0',
            'amazon.nova-lite-v1:0', 
            'amazon.nova-pro-v1:0'
        ]
        
        for model_id in test_models:
            print(f"\n🧪 Testing {model_id}...")
            
            # Nova request format (similar to Claude)
            body = json.dumps({
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": "Hello, can you respond with just 'Hello back!'?"}]
                    }
                ],
                "inferenceConfig": {
                    "max_new_tokens": 100,
                    "temperature": 0.0
                }
            })
            
            try:
                response = bedrock_runtime.invoke_model(
                    body=body,
                    modelId=model_id,
                    accept='application/json',
                    contentType='application/json'
                )
                
                response_body = json.loads(response['body'].read())
                if 'output' in response_body and 'message' in response_body['output']:
                    text_response = response_body['output']['message']['content'][0]['text']
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
        
        print("\n❌ No Nova models are accessible")
        return None
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == '__main__':
    print("🤖 Testing Amazon Nova Model Access")
    print("=" * 50)
    working_model = test_nova_models()
    
    if working_model:
        print(f"\n✅ Working Nova model found: {working_model}")
    else:
        print("\n❌ No working Nova models found")
