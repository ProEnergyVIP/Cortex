import argparse
from cortex.agent_builder import talk_to_agent_builder

def main():
    parser = argparse.ArgumentParser(description='Cortex CLI')
    subparsers = parser.add_subparsers(dest='command')
    
    # Add agent_builder subcommand
    subparsers.add_parser('agent_builder', help='Run the agent builder')
    
    args = parser.parse_args()
    
    if args.command == 'agent_builder':
        talk_to_agent_builder()
    else:
        parser.print_help()
