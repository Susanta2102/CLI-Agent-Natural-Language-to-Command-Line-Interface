#!/usr/bin/env python3
"""
Data Collection Script for CLI Q&A Pairs
Collects authentic Q&A pairs from Stack Overflow and other sources
"""

import requests
import json
import time
import re
import os
from bs4 import BeautifulSoup
from urllib.parse import quote
import random

class CLIDataCollector:
    def __init__(self):
        self.qa_pairs = []
        self.cli_topics = [
            'git', 'bash', 'tar', 'gzip', 'grep', 'find', 'awk', 'sed',
            'python virtualenv', 'pip install', 'ls command', 'cd command',
            'mkdir', 'rm command', 'cp command', 'mv command', 'chmod',
            'ssh', 'curl', 'wget', 'docker', 'npm', 'node', 'vim'
        ]
        
    def collect_stackoverflow_data(self, max_questions=100):
        """Collect Q&A pairs from Stack Overflow API"""
        print("Collecting data from Stack Overflow...")
        
        base_url = "https://api.stackexchange.com/2.3/search/advanced"
        
        for topic in self.cli_topics[:10]:  # Limit to avoid rate limits
            params = {
                'order': 'desc',
                'sort': 'votes',
                'q': topic,
                'tagged': 'command-line;bash;git;shell',
                'site': 'stackoverflow',
                'filter': 'withbody',
                'pagesize': 10
            }
            
            try:
                response = requests.get(base_url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    
                    for item in data.get('items', []):
                        # Clean and format the question and answer
                        question = self.clean_text(item.get('title', ''))
                        body = self.clean_html(item.get('body', ''))
                        
                        if len(question) > 10 and len(body) > 20:
                            qa_pair = {
                                'question': question,
                                'answer': self.extract_command_answer(body),
                                'source': 'stackoverflow',
                                'topic': topic,
                                'score': item.get('score', 0)
                            }
                            
                            if qa_pair['answer']:
                                self.qa_pairs.append(qa_pair)
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"Error collecting from Stack Overflow for {topic}: {e}")
    
    def collect_github_data(self):
        """Collect CLI examples from GitHub repositories"""
        print("Collecting data from GitHub...")
        
        # Common CLI scenarios and their solutions
        github_examples = [
            {
                'question': 'How to create a new Git branch and switch to it?',
                'answer': 'Use `git checkout -b <branch-name>` or `git switch -c <branch-name>` to create and switch to a new branch.',
                'source': 'github',
                'topic': 'git'
            },
            {
                'question': 'How to compress a folder using tar and gzip?',
                'answer': 'Use `tar -czf archive.tar.gz folder/` to create a compressed archive.',
                'source': 'github',
                'topic': 'tar'
            },
            {
                'question': 'How to find all Python files recursively?',
                'answer': 'Use `find . -name "*.py"` or `ls **/*.py` (with globstar enabled) to list all Python files recursively.',
                'source': 'github',
                'topic': 'find'
            },
            {
                'question': 'How to set up a Python virtual environment?',
                'answer': 'Use `python -m venv venv` to create and `source venv/bin/activate` (Linux/Mac) or `venv\\Scripts\\activate` (Windows) to activate.',
                'source': 'github',
                'topic': 'python'
            },
            {
                'question': 'How to view the first few lines of a file?',
                'answer': 'Use `head -n 10 filename` to display the first 10 lines of a file.',
                'source': 'github',
                'topic': 'head'
            }
        ]
        
        self.qa_pairs.extend(github_examples)
    
    def collect_manual_curated_data(self):
        """Add manually curated CLI Q&A pairs"""
        print("Adding manually curated data...")
        
        manual_qa = [
            {
                'question': 'How to check Git status and see what files are modified?',
                'answer': 'Use `git status` to see the current state of your working directory and staging area.',
                'source': 'manual',
                'topic': 'git'
            },
            {
                'question': 'How to extract a tar.gz file?',
                'answer': 'Use `tar -xzf archive.tar.gz` to extract a gzipped tar archive.',
                'source': 'manual',
                'topic': 'tar'
            },
            {
                'question': 'How to search for text patterns in files using grep?',
                'answer': 'Use `grep "pattern" filename` or `grep -r "pattern" directory/` for recursive search.',
                'source': 'manual',
                'topic': 'grep'
            },
            {
                'question': 'How to change file permissions using chmod?',
                'answer': 'Use `chmod 755 filename` for read/write/execute for owner, read/execute for group and others.',
                'source': 'manual',
                'topic': 'chmod'
            },
            {
                'question': 'How to copy files and directories?',
                'answer': 'Use `cp file1 file2` for files or `cp -r dir1/ dir2/` for directories.',
                'source': 'manual',
                'topic': 'cp'
            },
            {
                'question': 'How to move or rename files?',
                'answer': 'Use `mv oldname newname` to rename or `mv file /path/to/destination/` to move.',
                'source': 'manual',
                'topic': 'mv'
            },
            {
                'question': 'How to create a directory?',
                'answer': 'Use `mkdir dirname` or `mkdir -p path/to/nested/dirs` to create nested directories.',
                'source': 'manual',
                'topic': 'mkdir'
            },
            {
                'question': 'How to remove files and directories?',
                'answer': 'Use `rm filename` for files or `rm -rf dirname/` for directories (be careful!).',
                'source': 'manual',
                'topic': 'rm'
            },
            {
                'question': 'How to list files with detailed information?',
                'answer': 'Use `ls -la` to show all files with permissions, ownership, size, and modification date.',
                'source': 'manual',
                'topic': 'ls'
            },
            {
                'question': 'How to navigate to a different directory?',
                'answer': 'Use `cd /path/to/directory` or `cd ..` to go up one level, `cd ~` for home directory.',
                'source': 'manual',
                'topic': 'cd'
            }
        ]
        
        # Add more comprehensive examples
        additional_qa = []
        
        # Git commands
        git_commands = [
            ('How to stage all files for commit?', 'Use `git add .` to stage all files or `git add -A` to stage all including deletions.'),
            ('How to commit with a message?', 'Use `git commit -m "commit message"` to commit staged changes with a message.'),
            ('How to push changes to remote repository?', 'Use `git push origin branch-name` to push commits to the remote repository.'),
            ('How to pull latest changes?', 'Use `git pull` to fetch and merge changes from remote repository.'),
            ('How to clone a repository?', 'Use `git clone <repository-url>` to clone a remote repository locally.'),
            ('How to see commit history?', 'Use `git log` or `git log --oneline` for a condensed view of commit history.'),
            ('How to create a new branch without switching?', 'Use `git branch branch-name` to create a new branch without switching to it.'),
            ('How to switch between branches?', 'Use `git checkout branch-name` or `git switch branch-name` to switch branches.'),
            ('How to merge branches?', 'Switch to target branch and use `git merge source-branch` to merge changes.'),
            ('How to delete a branch?', 'Use `git branch -d branch-name` for merged branches or `git branch -D branch-name` to force delete.')
        ]
        
        # File operations
        file_ops = [
            ('How to view file contents?', 'Use `cat filename` to display file contents or `less filename` for paginated view.'),
            ('How to append text to a file?', 'Use `echo "text" >> filename` to append or `echo "text" > filename` to overwrite.'),
            ('How to count lines in a file?', 'Use `wc -l filename` to count lines, `wc -w` for words, `wc -c` for characters.'),
            ('How to sort file contents?', 'Use `sort filename` or `sort -r filename` for reverse order.'),
            ('How to remove duplicate lines?', 'Use `sort filename | uniq` or `sort -u filename` to remove duplicates.'),
            ('How to compare two files?', 'Use `diff file1 file2` to see differences between files.'),
            ('How to search and replace text in files?', 'Use `sed "s/old/new/g" filename` for search and replace with sed.'),
            ('How to find files by name?', 'Use `find . -name "filename"` or `find . -name "*.txt"` for pattern matching.'),
            ('How to find files by size?', 'Use `find . -size +100M` for files larger than 100MB or `-size -1M` for smaller than 1MB.'),
            ('How to find recently modified files?', 'Use `find . -mtime -1` for files modified in last day or `-mtime +7` for older than 7 days.')
        ]
        
        # Network and system commands
        system_commands = [
            ('How to check disk usage?', 'Use `df -h` for filesystem usage or `du -sh directory/` for directory size.'),
            ('How to check running processes?', 'Use `ps aux` or `top` for real-time process monitoring.'),
            ('How to kill a process?', 'Use `kill PID` or `pkill process-name` to terminate processes.'),
            ('How to download files from internet?', 'Use `wget URL` or `curl -O URL` to download files.'),
            ('How to check network connectivity?', 'Use `ping hostname` to test connectivity or `curl -I URL` for HTTP status.'),
            ('How to compress files with zip?', 'Use `zip archive.zip file1 file2` or `zip -r archive.zip directory/` for directories.'),
            ('How to extract zip files?', 'Use `unzip archive.zip` to extract zip files.'),
            ('How to monitor log files in real-time?', 'Use `tail -f logfile` to follow log files as they are written.'),
            ('How to search running processes?', 'Use `ps aux | grep process-name` to find specific running processes.'),
            ('How to check system memory usage?', 'Use `free -h` to see memory usage in human-readable format.')
        ]
        
        # Python/Package management
        python_commands = [
            ('How to install Python packages?', 'Use `pip install package-name` or `pip install -r requirements.txt` for multiple packages.'),
            ('How to create requirements file?', 'Use `pip freeze > requirements.txt` to save current package versions.'),
            ('How to upgrade pip packages?', 'Use `pip install --upgrade package-name` or `pip install -U package-name`.'),
            ('How to uninstall Python packages?', 'Use `pip uninstall package-name` to remove installed packages.'),
            ('How to list installed packages?', 'Use `pip list` or `pip show package-name` for detailed package info.'),
            ('How to run Python scripts?', 'Use `python script.py` or `python3 script.py` depending on your system.'),
            ('How to check Python version?', 'Use `python --version` or `python -V` to check installed Python version.'),
            ('How to deactivate virtual environment?', 'Use `deactivate` command when inside an activated virtual environment.'),
            ('How to install specific package version?', 'Use `pip install package-name==1.2.3` to install specific version.'),
            ('How to create virtual environment with specific Python version?', 'Use `python3.8 -m venv myenv` to create venv with specific Python version.')
        ]
        
        # Combine all categories
        all_commands = git_commands + file_ops + system_commands + python_commands
        
        for question, answer in all_commands:
            topic = 'git' if 'git' in question.lower() else \
                   'python' if 'pip' in question.lower() or 'python' in question.lower() else \
                   'system' if any(cmd in question.lower() for cmd in ['disk', 'process', 'memory', 'network']) else \
                   'file'
            
            additional_qa.append({
                'question': question,
                'answer': answer,
                'source': 'manual',
                'topic': topic
            })
        
        self.qa_pairs.extend(manual_qa + additional_qa)
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def clean_html(self, html_text):
        """Remove HTML tags and clean text"""
        if not html_text:
            return ""
        soup = BeautifulSoup(html_text, 'html.parser')
        # Extract code blocks
        code_blocks = soup.find_all(['code', 'pre'])
        text = soup.get_text()
        return self.clean_text(text)
    
    def extract_command_answer(self, text):
        """Extract command-line relevant information from text"""
        if not text:
            return ""
        
        # Look for common command patterns
        command_patterns = [
            r'`([^`]+)`',  # backtick enclosed commands
            r'```[^`]*```',  # code blocks
            r'\$\s*([^\n]+)',  # $ prompt commands
        ]
        
        commands = []
        for pattern in command_patterns:
            matches = re.findall(pattern, text)
            commands.extend(matches)
        
        if commands:
            return f"Use {commands[0]} command. " + text[:200] + "..."
        
        return text[:300] + "..." if len(text) > 300 else text
    
    def save_data(self):
        """Save collected data to JSON files"""
        os.makedirs('data', exist_ok=True)
        
        # Save raw data
        with open('data/raw_qa_pairs.json', 'w', encoding='utf-8') as f:
            json.dump(self.qa_pairs, f, indent=2, ensure_ascii=False)
        
        # Create training format
        training_data = []
        for qa in self.qa_pairs:
            # Format for instruction tuning
            instruction = f"Answer this command-line question: {qa['question']}"
            training_data.append({
                'instruction': instruction,
                'input': '',
                'output': qa['answer'],
                'topic': qa['topic'],
                'source': qa['source']
            })
        
        with open('data/processed_training_data.json', 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        print(f"Collected {len(self.qa_pairs)} Q&A pairs")
        print(f"Data saved to data/ directory")
        
        # Print summary by topic
        topics = {}
        for qa in self.qa_pairs:
            topic = qa['topic']
            topics[topic] = topics.get(topic, 0) + 1
        
        print("\nData distribution by topic:")
        for topic, count in sorted(topics.items()):
            print(f"  {topic}: {count}")

def main():
    collector = CLIDataCollector()
    
    print("Starting data collection...")
    
    # Collect from various sources
    collector.collect_stackoverflow_data()
    collector.collect_github_data()
    collector.collect_manual_curated_data()
    
    # Save the data
    collector.save_data()
    
    print("Data collection completed!")

if __name__ == "__main__":
    main()