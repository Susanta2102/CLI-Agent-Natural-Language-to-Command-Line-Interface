#!/usr/bin/env python3
"""
Evaluation Script for CLI Agent
Performs static and dynamic evaluation with BLEU/ROUGE-L metrics
"""

import os
import json
import sys
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, base_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", adapter_path="training/adapters"):
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Test prompts from the task requirements
        self.test_prompts = [
            "Create a new Git branch and switch to it.",
            "Compress the folder reports into reports.tar.gz.",
            "List all Python files in the current directory recursively.",
            "Set up a virtual environment and install requests.",
            "Fetch only the first ten lines of a file named output.log.",
            # Two additional edge cases
            "Remove all .pyc files from the project and ignore them in git.",
            "Search for TODO comments in all JavaScript files and show line numbers."
        ]
        
        # Reference answers for evaluation
        self.reference_answers = [
            "Use git checkout -b <branch-name> or git switch -c <branch-name> to create and switch to a new branch.",
            "Use tar -czf reports.tar.gz reports/ to create a compressed archive of the reports folder.",
            "Use find . -name '*.py' to list all Python files recursively in the current directory.",
            "Use python -m venv venv to create and source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows) to activate, then pip install requests.",
            "Use head -n 10 output.log to display the first ten lines of the file.",
            "Use find . -name '*.pyc' -delete to remove .pyc files and echo '*.pyc' >> .gitignore to ignore them in git.",
            "Use grep -n 'TODO' **/*.js or find . -name '*.js' -exec grep -n 'TODO' {} + to search for TODO comments with line numbers."
        ]
        
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
        
    def load_models(self):
        """Load both base and fine-tuned models"""
        logger.info("Loading models for comparison...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Load fine-tuned model
        if os.path.exists(self.adapter_path):
            self.fine_tuned_model = PeftModel.from_pretrained(self.base_model, self.adapter_path)
            logger.info("Fine-tuned model loaded successfully")
        else:
            logger.warning("Fine-tuned model not found, using base model for both")
            self.fine_tuned_model = self.base_model
    
    def generate_response(self, model, prompt):
        """Generate response using the specified model"""
        try:
            # Format prompt
            formatted_prompt = f"### Instruction:\nAnswer this command-line question: {prompt}\n\n### Response:\n"
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = response[len(formatted_prompt):].strip()
            
            return generated
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Error generating response"
    
    def calculate_bleu_score(self, reference, candidate):
        """Calculate BLEU score"""
        try:
            reference_tokens = reference.lower().split()
            candidate_tokens = candidate.lower().split()
            
            # Calculate BLEU-4 score
            score = sentence_bleu(
                [reference_tokens],
                candidate_tokens,
                smoothing_function=self.smoothing
            )
            return score
        except:
            return 0.0
    
    def calculate_rouge_scores(self, reference, candidate):
        """Calculate ROUGE scores"""
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def score_plan_quality(self, response, prompt):
        """Score plan quality on 0-2 scale"""
        response_lower = response.lower()
        prompt_lower = prompt.lower()
        
        score = 0
        
        # Check for relevant keywords and commands
        if "git" in prompt_lower and "branch" in prompt_lower:
            if any(cmd in response_lower for cmd in ["git checkout", "git switch", "git branch"]):
                score += 1
            if "git checkout -b" in response_lower or "git switch -c" in response_lower:
                score += 1
                
        elif "compress" in prompt_lower and "tar" in prompt_lower:
            if "tar" in response_lower and ("-c" in response_lower or "create" in response_lower):
                score += 1
            if "tar -czf" in response_lower or "gzip" in response_lower:
                score += 1
                
        elif "python files" in prompt_lower and "recursive" in prompt_lower:
            if "find" in response_lower and "*.py" in response_lower:
                score += 1
            if "find . -name" in response_lower:
                score += 1
                
        elif "virtual environment" in prompt_lower:
            if any(cmd in response_lower for cmd in ["venv", "virtualenv", "python -m venv"]):
                score += 1
            if "activate" in response_lower and "pip install" in response_lower:
                score += 1
                
        elif "first" in prompt_lower and "lines" in prompt_lower:
            if "head" in response_lower:
                score += 1
            if "head -n" in response_lower or "head -10" in response_lower:
                score += 1
                
        elif ".pyc" in prompt_lower and "remove" in prompt_lower:
            if "find" in response_lower and "delete" in response_lower:
                score += 1
            if "gitignore" in response_lower or ".gitignore" in response_lower:
                score += 1
                
        elif "todo" in prompt_lower and "grep" in prompt_lower:
            if "grep" in response_lower:
                score += 1
            if "grep -n" in response_lower or "line number" in response_lower:
                score += 1
        
        else:
            # Generic scoring for other prompts
            if len(response) > 20 and any(word in response_lower for word in ["use", "command", "run"]):
                score += 1
            if any(char in response for char in ['-', '|', '>']):  # Command-like syntax
                score += 1
        
        return min(score, 2)  # Cap at 2
    
    def run_static_evaluation(self):
        """Run static evaluation comparing base vs fine-tuned outputs"""
        logger.info("Running static evaluation...")
        
        self.load_models()
        
        results = []
        
        for i, (prompt, reference) in enumerate(zip(self.test_prompts, self.reference_answers)):
            logger.info(f"Evaluating prompt {i+1}: {prompt[:50]}...")
            
            # Generate responses
            base_response = self.generate_response(self.base_model, prompt)
            ft_response = self.generate_response(self.fine_tuned_model, prompt)
            
            # Calculate metrics
            base_bleu = self.calculate_bleu_score(reference, base_response)
            ft_bleu = self.calculate_bleu_score(reference, ft_response)
            
            base_rouge = self.calculate_rouge_scores(reference, base_response)
            ft_rouge = self.calculate_rouge_scores(reference, ft_response)
            
            base_quality = self.score_plan_quality(base_response, prompt)
            ft_quality = self.score_plan_quality(ft_response, prompt)
            
            result = {
                'prompt': prompt,
                'reference': reference,
                'base_response': base_response,
                'fine_tuned_response': ft_response,
                'metrics': {
                    'base_bleu': base_bleu,
                    'fine_tuned_bleu': ft_bleu,
                    'base_rouge': base_rouge,
                    'fine_tuned_rouge': ft_rouge,
                    'base_quality_score': base_quality,
                    'fine_tuned_quality_score': ft_quality
                }
            }
            
            results.append(result)
        
        # Calculate averages
        avg_metrics = {
            'base_bleu': sum(r['metrics']['base_bleu'] for r in results) / len(results),
            'fine_tuned_bleu': sum(r['metrics']['fine_tuned_bleu'] for r in results) / len(results),
            'base_rougeL': sum(r['metrics']['base_rouge']['rougeL'] for r in results) / len(results),
            'fine_tuned_rougeL': sum(r['metrics']['fine_tuned_rouge']['rougeL'] for r in results) / len(results),
            'base_quality': sum(r['metrics']['base_quality_score'] for r in results) / len(results),
            'fine_tuned_quality': sum(r['metrics']['fine_tuned_quality_score'] for r in results) / len(results)
        }
        
        # Save results
        with open("eval_static.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
                
        
        # Generate markdown report
        self.generate_static_report(results, avg_metrics)
        
        logger.info("Static evaluation completed")
        return results, avg_metrics
    
    def generate_static_report(self, results, avg_metrics):
        """Generate static evaluation markdown report"""
        markdown_content = f"""# Static Evaluation Report

## Overview
Comparison of base model vs fine-tuned model responses on {len(results)} test prompts.

## Average Metrics
| Metric | Base Model | Fine-tuned Model | Improvement |
|--------|------------|------------------|-------------|
| BLEU Score | {avg_metrics['base_bleu']:.3f} | {avg_metrics['fine_tuned_bleu']:.3f} | {(avg_metrics['fine_tuned_bleu'] - avg_metrics['base_bleu']):.3f} |
| ROUGE-L | {avg_metrics['base_rougeL']:.3f} | {avg_metrics['fine_tuned_rougeL']:.3f} | {(avg_metrics['fine_tuned_rougeL'] - avg_metrics['base_rougeL']):.3f} |
| Quality Score | {avg_metrics['base_quality']:.3f} | {avg_metrics['fine_tuned_quality']:.3f} | {(avg_metrics['fine_tuned_quality'] - avg_metrics['base_quality']):.3f} |

## Individual Results

"""
        
        for i, result in enumerate(results, 1):
            markdown_content += f"""### Test {i}: {result['prompt']}

**Reference Answer:**
{result['reference']}

**Base Model Response:**
{result['base_response']}

**Fine-tuned Model Response:**
{result['fine_tuned_response']}

**Metrics:**
- BLEU: Base {result['metrics']['base_bleu']:.3f} → Fine-tuned {result['metrics']['fine_tuned_bleu']:.3f}
- ROUGE-L: Base {result['metrics']['base_rouge']['rougeL']:.3f} → Fine-tuned {result['metrics']['fine_tuned_rouge']['rougeL']:.3f}
- Quality: Base {result['metrics']['base_quality_score']}/2 → Fine-tuned {result['metrics']['fine_tuned_quality_score']}/2

---

"""
        
        with open("eval_static.md", "w", encoding="utf-8") as f:
            f.write(markdown_content)
    
    def run_dynamic_evaluation(self):
        """Run dynamic evaluation using the CLI agent"""
        logger.info("Running dynamic evaluation...")
        
        # Import and use the CLI agent
        from agent import CLIAgent
        
        agent = CLIAgent()
        
        results = []
        
        for prompt in self.test_prompts:
            logger.info(f"Testing agent with: {prompt[:50]}...")
            
            try:
                result = agent.process_instruction(prompt)
                
                # Score the plan quality
                quality_score = self.score_plan_quality(' '.join(result['plan']), prompt)
                
                # Count extracted commands
                command_count = len(result['commands'])
                
                agent_result = {
                    'prompt': prompt,
                    'plan': result['plan'],
                    'commands': result['commands'],
                    'command_count': command_count,
                    'quality_score': quality_score,
                    'execution_successful': len(result['execution_log']) > 0
                }
                
                results.append(agent_result)
                
            except Exception as e:
                logger.error(f"Error in dynamic evaluation: {e}")
                results.append({
                    'prompt': prompt,
                    'plan': [],
                    'commands': [],
                    'command_count': 0,
                    'quality_score': 0,
                    'execution_successful': False,
                    'error': str(e)
                })
        
        # Generate dynamic report
        self.generate_dynamic_report(results)
        
        logger.info("Dynamic evaluation completed")
        return results
    
    def generate_dynamic_report(self, results):
        """Generate dynamic evaluation markdown report"""
        avg_quality = sum(r['quality_score'] for r in results) / len(results)
        avg_commands = sum(r['command_count'] for r in results) / len(results)
        success_rate = sum(1 for r in results if r['execution_successful']) / len(results)
        
        markdown_content = f"""# Dynamic Evaluation Report

## Overview
Evaluation of CLI Agent performance on {len(results)} test prompts.

## Summary Metrics
- Average Quality Score: {avg_quality:.2f}/2
- Average Commands per Plan: {avg_commands:.1f}
- Execution Success Rate: {success_rate:.1%}

## Agent Performance Scoring

| Test | Prompt | Commands | Quality Score | Status |
|------|--------|----------|---------------|---------|
"""
        
        for i, result in enumerate(results, 1):
            status = "✅ Success" if result['execution_successful'] else "❌ Failed"
            markdown_content += f"| {i} | {result['prompt'][:40]}... | {result['command_count']} | {result['quality_score']}/2 | {status} |\n"
        
        markdown_content += f"""

## Detailed Results

"""
        
        for i, result in enumerate(results, 1):
            markdown_content += f"""### Test {i}: {result['prompt']}

**Generated Plan:**
"""
            for j, step in enumerate(result['plan'], 1):
                markdown_content += f"{j}. {step}\n"
            
            markdown_content += f"""
**Extracted Commands ({result['command_count']}):**
"""
            for cmd in result['commands']:
                markdown_content += f"- `{cmd}`\n"
            
            markdown_content += f"""
**Quality Score:** {result['quality_score']}/2
**Execution:** {"Successful" if result['execution_successful'] else "Failed"}

---

"""
        
        with open("eval_dynamic.md", "w", encoding="utf-8") as f:
            f.write(markdown_content)

def main():
    
    
    """Main evaluation function"""
    evaluator = ModelEvaluator()
    
    print("🔍 Starting evaluation...")
    
    # Run static evaluation
    print("📊 Running static evaluation...")
    static_results, avg_metrics = evaluator.run_static_evaluation()
    
    # Run dynamic evaluation
    print("🤖 Running dynamic evaluation...")
    dynamic_results = evaluator.run_dynamic_evaluation()
    
    print("✅ Evaluation completed!")
    print(f"📄 Reports generated:")
    print(f"  - eval_static.md")
    print(f"  - eval_dynamic.md")
    print(f"  - eval_static.json")

if __name__ == "__main__":
    main()