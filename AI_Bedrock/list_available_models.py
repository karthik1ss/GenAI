#!/usr/bin/env python3

import json
import boto3
from botocore.exceptions import ClientError

def list_bedrock_models():
    try:
        session = boto3.Session()
        bedrock = session.client(service_name='bedrock', region_name='us-west-2')
        
        response = bedrock.list_foundation_models()
        models = response['modelSummaries']
        
        print(f"Found {len(models)} available models:\n")
        
        # Group by provider
        providers = {}
        for model in models:
            provider = model['providerName']
            if provider not in providers:
                providers[provider] = []
            providers[provider].append(model)
        
        for provider, provider_models in providers.items():
            print(f"🏢 {provider}:")
            for model in provider_models:
                status = model['modelLifecycle']['status']
                modalities = f"Input: {', '.join(model['inputModalities'])} | Output: {', '.join(model['outputModalities'])}"
                print(f"  ✅ {model['modelName']}")
                print(f"     ID: {model['modelId']}")
                print(f"     Status: {status}")
                print(f"     Modalities: {modalities}")
                print()
        
        # Test access to a few common models
        print("\n🔍 Testing model access...")
        bedrock_runtime = session.client(service_name='bedrock-runtime', region_name='us-west-2')
        
        test_models = [
            'amazon.titan-tg1-large',
             'anthropic.claude-3-5-sonnet-20241022-v2:0',
            'us.anthropic.claude-3-5-sonnet-20241022-v2:0',
            "us.anthropic.claude-sonnet-4-20250514-v1:0",
            'amazon.titan-text-lite-v1',
        ]
        
        accessible_models = []
        
        for model_id in test_models:
            try:
                # Try a simple test request
                body = json.dumps({
                    'inputText': 'Hello',
                    'textGenerationConfig': {
                        'temperature': 0.0,
                        'topP': 1.0,
                        'maxTokenCount': 10
                    }
                })
                
                # This will fail if we don't have access
                bedrock_runtime.invoke_model(
                    body=body,
                    modelId=model_id,
                    accept='application/json',
                    contentType='application/json'
                )
                accessible_models.append(model_id)
                print(f"  ✅ {model_id} - ACCESSIBLE")
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'AccessDeniedException':
                    print(f"  ❌ {model_id} - ACCESS DENIED")
                elif error_code == 'ValidationException':
                    # This might mean the model exists but our request format is wrong
                    print(f"  ⚠️  {model_id} - VALIDATION ERROR (might be accessible with correct format)")
                else:
                    print(f"  ❓ {model_id} - ERROR: {error_code}")
            except Exception as e:
                print(f"  ❓ {model_id} - UNKNOWN ERROR: {str(e)}")
        
        print(f"\n✅ Accessible models: {accessible_models}")
        return accessible_models
        
    except Exception as e:
        print(f"Error: {e}")
        return []

if __name__ == '__main__':
    list_bedrock_models()
