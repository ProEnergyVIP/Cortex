import boto3
import json

from intellifun.message import AIMessage


endpoint_name = 'jumpstart-dft-meta-textgeneration-llama-2-7b-f'

aws_client = None

def get_client():
    global aws_client

    if aws_client is None:
        session = boto3.Session(region_name='us-west-2')
        aws_client = session.client('sagemaker-runtime')
    
    return aws_client


class Llama2Chat:
    def __init__(self, temperature: float = 0.5):
        self.client = get_client()
        self.temperature = temperature

    def call(self, messages):
        msgs = [m.to_dict() for m in messages]
        json_body = {
            'inputs': [msgs],
            'parameters': {'max_new_tokens': 2000, 'top_p': 0.9, 'temperature': self.temperature}
        }

        try:
            response = self.client.invoke_endpoint(
                EndpointName=endpoint_name,
                Body=json.dumps(json_body),
                ContentType='application/json',
                Accept='application/json',
                CustomAttributes='accept_eula=true'
            )
        except Exception as e:
            print(e)

        res = json.loads(response['Body'].read().decode("utf-8"))
        res_msg = res[0]['generation']['content']
        return AIMessage(content=res_msg)
