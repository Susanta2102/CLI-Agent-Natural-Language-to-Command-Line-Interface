#!/usr/bin/env python3
"""
CLI Agent - Natural Language to Command Line Interface
Accepts natural language instructions and generates executable command plans
"""

import sys
import os
import json
import re
import datetime
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CLIAgent:
    def __init__(self, base_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", adapter_path="training/adapters"):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.log_file = "logs/trace.jsonl"
        
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned model with LoRA adapter or fallback to base model"""
        try:
            logger.info(f"Loading base model: {self.base_model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                trust_remote_code=True,
                padding_side="right"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # CPU-friendly model loading
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.float16,
                device_map=None if not torch.cuda.is_available() else "auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Try to load the fine-tuned adapter
            if os.path.exists(self.adapter_path) and any(Path(self.adapter_path).iterdir()):
                try:
                    logger.info(f"Loading LoRA adapter from: {self.adapter_path}")
                    self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
                    logger.info("✅ Fine-tuned model loaded successfully")
                    self.model_type = "fine_tuned"
                except Exception as e:
                    logger.warning(f"Failed to load adapter: {e}, using base model")
                    self.model = base_model
                    self.model_type = "base_model"
            else:
                logger.warning(f"No adapter found at {self.adapter_path}, using base model")
                self.model = base_model
                self.model_type = "base_model"
                
            # Move to device
            self.model = self.model.to(self.device)
            logger.info(f"Model loaded on {self.device} ({self.model_type})")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Falling back to rule-based responses")
            self.model = None
            self.tokenizer = None
            self.model_type = "fallback"
    
    def generate_plan(self, instruction):
        """Generate a step-by-step plan for the given instruction"""
        if self.model is None:
            return self.fallback_plan_generation(instruction)
        
        try:
            # Format prompt for instruction following
            prompt = f"### Instruction:\nAnswer this command-line question: {instruction}\n\n### Response:\n"
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the generated part (after the prompt)
            generated_text = response[len(prompt):].strip()
            
            # Post-process the response to create a step-by-step plan
            plan = self.format_as_plan(instruction, generated_text)
            
            return plan
            
        except Exception as e:
            logger.error(f"Error generating plan with model: {e}")
            return self.fallback_plan_generation(instruction)
    
    def fallback_plan_generation(self, instruction):
        """Rule-based plan generation as fallback"""
        logger.info("Using fallback plan generation")
        
        instruction_lower = instruction.lower()
        
        # Git operations
        if "git" in instruction_lower and "branch" in instruction_lower and "switch" in instruction_lower:
            return [
                "Check current Git status",
                "git status",
                "Create and switch to new branch",
                "git checkout -b <branch-name>",
                "Verify you're on the new branch",
                "git branch"
            ]
        
        # Compression operations
        elif "compress" in instruction_lower and ("tar" in instruction_lower or "gz" in instruction_lower):
            return [
                "Navigate to the parent directory of the folder to compress",
                "cd /path/to/parent/directory",
                "Create compressed archive",
                "tar -czf archive.tar.gz folder_name/",
                "Verify the archive was created",
                "ls -la *.tar.gz"
            ]
        
        # Find Python files
        elif "python" in instruction_lower and "files" in instruction_lower and "recursive" in instruction_lower:
            return [
                "Use find command to locate all Python files",
                "find . -name \"*.py\"",
                "Alternative: use ls with globstar (if enabled)",
                "ls **/*.py",
                "Count the number of Python files found",
                "find . -name \"*.py\" | wc -l"
            ]
        
        # Virtual environment
        elif "virtual" in instruction_lower and "environment" in instruction_lower:
            return [
                "Create a new virtual environment",
                "python -m venv venv",
                "Activate the virtual environment (Windows)",
                "venv\\Scripts\\activate",
                "Activate the virtual environment (Linux/Mac)",
                "source venv/bin/activate",
                "Install requests package",
                "pip install requests",
                "Verify installation",
                "pip list"
            ]
        
        # Read file lines
        elif "first" in instruction_lower and "lines" in instruction_lower:
            return [
                "Use head command to display first lines of file",
                "head -n 10 output.log",
                "Alternative: use sed for more control",
                "sed -n '1,10p' output.log",
                "View file details",
                "ls -la output.log"
            ]
        
        # Generic plan
        else:
            return [
                f"Analyze the instruction: {instruction}",
                "# Determine the appropriate command-line tools needed",
                "Break down the task into steps",
                "# Execute each step carefully",
                "Verify the results",
                "# Check that the task was completed successfully"
            ]
    
    def format_as_plan(self, instruction, generated_text):
        """Format the generated text as a step-by-step plan"""
        lines = generated_text.split('\n')
        plan = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 3:  # Skip very short lines
                # Clean up the line
                if line.startswith('-') or line.startswith('*'):
                    line = line[1:].strip()
                elif line.startswith(('1.', '2.', '3.', '4.', '5.')):
                    line = re.sub(r'^\d+\.\s*', '', line)
                
                plan.append(line)
        
        # If no good plan was extracted, create a basic one
        if len(plan) < 2:
            plan = [
                f"Task: {instruction}",
                "Analyze the requirements",
                "Execute the appropriate command",
                "Verify the results"
            ]
        
        return plan
    
    def extract_commands(self, plan):
        """Extract shell commands from the plan"""
        commands = []
        
        # Common command patterns
        command_patterns = [
            r'^(git|ls|cd|mkdir|rm|cp|mv|find|grep|tar|head|tail|cat|echo|pip|python)\s+.*',
            r'^[a-zA-Z_][a-zA-Z0-9_-]*\s+.*',  # General command pattern
            r'`([^`]+)`',  # Commands in backticks
        ]
        
        for step in plan:
            # Check if the step looks like a command
            for pattern in command_patterns:
                if re.match(pattern, step.strip(), re.IGNORECASE):
                    # Clean the command
                    cmd = step.strip()
                    # Remove backticks if present
                    cmd = re.sub(r'`([^`]+)`', r'\1', cmd)
                    commands.append(cmd)
                    break
        
        return commands
    
    def execute_dry_run(self, commands):
        """Execute commands in dry-run mode (echo only)"""
        execution_log = []
        
        for cmd in commands:
            # Skip comments and explanations
            if cmd.startswith('#') or not cmd.strip():
                continue
                
            # Log the dry-run execution
            dry_run_cmd = f"echo '{cmd}'"
            execution_log.append({
                'original_command': cmd,
                'dry_run_command': dry_run_cmd,
                'execution_type': 'dry_run',
                'timestamp': datetime.datetime.now().isoformat()
            })
            
            # Print the dry-run
            print(f"[DRY RUN] {cmd}")
        
        return execution_log
    
    def log_trace(self, instruction, plan, commands, execution_log):
        """Log the complete trace to JSONL file"""
        trace_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'instruction': instruction,
            'plan': plan,
            'extracted_commands': commands,
            'execution_log': execution_log,
            'model_used': getattr(self, 'model_type', 'unknown') if self.model else 'fallback',
            'device': str(self.device),
            'adapter_path': self.adapter_path
        }
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(trace_entry) + '\n')
        
        logger.info(f"Trace logged to {self.log_file}")
    
    def process_instruction(self, instruction):
        """Main method to process a natural language instruction"""
        print(f"\n🤖 Processing instruction: {instruction}")
        print("=" * 60)
        
        # Generate plan
        print("📋 Generating step-by-step plan...")
        plan = self.generate_plan(instruction)
        
        print("\n📝 Plan:")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step}")
        
        # Extract commands
        commands = self.extract_commands(plan)
        
        print(f"\n⚙️  Extracted commands ({len(commands)}):")
        for cmd in commands:
            print(f"  → {cmd}")
        
        # Execute in dry-run mode
        print(f"\n🧪 Dry-run execution:")
        execution_log = self.execute_dry_run(commands)
        
        # Log everything
        self.log_trace(instruction, plan, commands, execution_log)
        
        print(f"\n✅ Task completed. Trace logged to {self.log_file}")
        
        return {
            'plan': plan,
            'commands': commands,
            'execution_log': execution_log
        }

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='CLI Agent - Natural Language to Command Line')
    parser.add_argument('instruction', nargs='?', help='Natural language instruction')
    parser.add_argument('--adapter-path', default='training/adapters', help='Path to LoRA adapter')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Initialize agent
    print("🚀 Initializing CLI Agent...")
    agent = CLIAgent(adapter_path=args.adapter_path)
    
    if args.interactive:
        print("💬 Interactive mode. Type 'quit' or 'exit' to stop.")
        while True:
            try:
                instruction = input("\n📝 Enter instruction: ").strip()
                if instruction.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                if instruction:
                    agent.process_instruction(instruction)
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    elif args.instruction:
        agent.process_instruction(args.instruction)
    
    else:
        print("❌ Please provide an instruction or use --interactive mode")
        print("Example: python agent.py \"Create a new Git branch and switch to it\"")

if __name__ == "__main__":
    main()