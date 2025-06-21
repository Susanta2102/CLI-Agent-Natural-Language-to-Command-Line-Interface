# Static Evaluation Report

## Overview
Comparison of base model vs fine-tuned model responses on 7 test prompts.

## Average Metrics
| Metric | Base Model | Fine-tuned Model | Improvement |
|--------|------------|------------------|-------------|
| BLEU Score | 0.014 | 0.017 | 0.003 |
| ROUGE-L | 0.130 | 0.128 | -0.001 |
| Quality Score | 1.000 | 0.857 | -0.143 |

## Individual Results

### Test 1: Create a new Git branch and switch to it.

**Reference Answer:**
Use git checkout -b <branch-name> or git switch -c <branch-name> to create and switch to a new branch.

**Base Model Response:**
Use git log to see commit history. How to see commit details with a click of a button. The command is `git log --oneline`. I want to run this command directly from the command line. How to move to a different branch then switch back. The command is: git checkout branch-name /* I want to run this without typing the name of the branch. How to merge changes from a remote branch to a local one. The command is: git merge remote-branch /* I want to see a list of all outstanding commits. How to do this without using the graph tool. The command is: git log --oneline -1 // this is a long-running command. I want to run it quickly. What is the command to view commit details with a click of a button? I want to run this directly from the command line. How to merge changes from a remote branch to a local one. The command is: git merge remote-branch /* I want to

**Fine-tuned Model Response:**
Use `git checkout -b <branch-name>` or `git switch -c <branch-name>` to create and switch to a new branch. I want to create a new branch without creating a new command line window. How to do this? I want to switch to a new branch already named in my current command line window. How to use `git checkout -b <branch-name>` or `git switch -c <branch-name>` to create and switch to a new branch? I have recently started using Git and I am trying to automate a process that I have been doing by hand. I want to create a command that creates a new branch without using a new command line window. I have found this command that I use to create a new command line window: $ windows2100 --mount volume=volumename /mnt/volumename/git/cmd/git-clone.cmd /v/c/User/Music/Git/

**Metrics:**
- BLEU: Base 0.004 → Fine-tuned 0.061
- ROUGE-L: Base 0.118 → Fine-tuned 0.241
- Quality: Base 1/2 → Fine-tuned 2/2

---

### Test 2: Compress the folder reports into reports.tar.gz.

**Reference Answer:**
Use tar -czf reports.tar.gz reports/ to create a compressed archive of the reports folder.

**Base Model Response:**
Use `tar -czf reports.tar.gz reports` to create a compressed archive. command. I want to run this command: run `tar -czf reports.tar.gz reports` to create a compressed archive. I want to delete files from the archive. How to do this? response. I need to run this command: deactivate; disktree /mnt/<path-to-mounted-drive> -d -T "folder name" archive.txt

### Response:
cd /mnt/<path-to-mounted-drive> // Change directory to mounted drive. tape -r /mnt/<path-to-mounted-drive> // Create a new archive with repeatability. command. I want to use this command: cat archive.txt >> reports.tar.gz // Merge files into archive. How to extract an archive on a different drive than the one from which

**Fine-tuned Model Response:**
Use the `gzip -c report.txt > reports.tar.gz` command to compress a text file. How to extract a tar.gz file? What is the right command? Get there! This is a command-line question. I want to extract a tar.gz file that I have downloaded. I know the right command to download a file: git clone url-of-the-repository command. I am trying to ask my system to extract a gzipped file. I want to do this from a command line. I know the right command to download a file: git clone url-of-the-repository command. I am trying to ask my system to extract a gzipped file. What is the right command to do this from a command line? Maybe something like this: $ gzip -d report.tar.gz  # or something like this: $ tar -xzf report.tar.gz  # what is the correct command to

**Metrics:**
- BLEU: Base 0.028 → Fine-tuned 0.002
- ROUGE-L: Base 0.207 → Fine-tuned 0.110
- Quality: Base 2/2 → Fine-tuned 2/2

---

### Test 3: List all Python files in the current directory recursively.

**Reference Answer:**
Use find . -name '*.py' to list all Python files recursively in the current directory.

**Base Model Response:**
Use `ls **/*.py` will list all Python files recursively, including subdirectories. But `ls *py` will list all Python files with name wildcard. I want to delete just the Python files with name wildcard (*py). How to do that using git command? I know that git has a `git delete-branch` command to remove a branch, and `git checkout branch-name` to switch to a specific branch. I'm wondering if there is a way to do things like this with the command line. I want to delete a specific Python file with a wildcard, like this: git-upload-pack --delete **/*.py As I'm not sure what the correct command is for that, I try: git-upload-pack --delete **/*.py Untarably, this is trying to move all Python files with name wildcard to a separate directory. How can I change or delete files specifically? I

**Fine-tuned Model Response:**
Use `ls **/*.py` will list all Python files recursively, including nested directories. command. I want to extract a list of Python files with recurring patterns. I want to run a command that lists all Python files in the current directory recursively, with any matching patterns specified on the command line. I know that I can use a shell script to do this, but I'm looking for a command-line tool. I have recently installed git-upload-pack, and it contains a script, `uploadpct.sh`, that does just this: $ uploadpct.sh "list all Python files with recurring patterns" echo "Running `ls **/*.py` will list all Python files recursively, including nested directories.". command. I've modified it to include any pattern specs on the command line, although I'm not sure if this is possible with a shell script. echo "list all Python files with any matching

**Metrics:**
- BLEU: Base 0.019 → Fine-tuned 0.025
- ROUGE-L: Base 0.113 → Fine-tuned 0.138
- Quality: Base 0/2 → Fine-tuned 0/2

---

### Test 4: Set up a virtual environment and install requests.

**Reference Answer:**
Use python -m venv venv to create and source venv/bin/activate (Linux/Mac) or venv\Scripts\activate (Windows) to activate, then pip install requests.

**Base Model Response:**
Use --system-site-packages to install packages locally. command. I am trying to install specific versions of packages using `pip install package-name == 1.2.3`. I ... system path, so I use --system-site-packages to do this: $ pip install -U --system-site-packages package-name == 1.2.3 But when I look at the directory structure of the virtual environment, I see that there is a files directory and a __pycache__ directory. How do I make pip use this directory structure? I have read the answer to the same question on Stack Exchange, but there is something I am not understanding correctly: $ pip install -U --no-site-packages package-name == 1.2.3 This command is working correctly, but it is installing packages globally, rather than locally. I know that I can install specific versions with `pip install package-name == 1.2.3

**Fine-tuned Model Response:**
Use --system-site-packages to install packages system-wide, or --user to install packages for a specific user. How to check running processes in Python? I want to run a script that checks running processes in Python. I know I can use psutil to do this, but is there an easy way to do this in Python? How to copy files from one system to another? I want to copy a file from an Ubuntu system to a Raspberry Pi system. Is this possible? I know there are shell commands for this, but I'm looking for an approach in Python. How to create a directory if it doesn't exist in Python. I want to create a directory if it doesn't exist, even if that means running a command. Is this possible in Python? How to delete a directory and all its contents in Python. I want to delete a directory and all its contents recursively. Is this possible in Python? I want to delete a directory and

**Metrics:**
- BLEU: Base 0.004 → Fine-tuned 0.003
- ROUGE-L: Base 0.071 → Fine-tuned 0.072
- Quality: Base 0/2 → Fine-tuned 0/2

---

### Test 5: Fetch only the first ten lines of a file named output.log.

**Reference Answer:**
Use head -n 10 output.log to display the first ten lines of the file.

**Base Model Response:**
Use git log --format=%B -10 /home/me/logfile.log | head -n 10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

**Fine-tuned Model Response:**
Use git log --name-status -10F output.log to do this or git log -10nF output.log to see just the commit message. command. I want to extract just the commit message from an entire log file. I know I can see the first ten lines with `fetch -x log --name-status -10F output.log` but how do I do this for a specific file? When I use git log --name-status -10F output.log I get the full commit message for each line. How can I do this for just the commit message? I want to run this command: $ git log --name-status -10F output.log How can I do this for a specific file? I know I can see the first ten lines with `fetch -x log --name-status -10F output.log` but how do I do this for a specific file? When I use git log

**Metrics:**
- BLEU: Base 0.033 → Fine-tuned 0.021
- ROUGE-L: Base 0.214 → Fine-tuned 0.122
- Quality: Base 2/2 → Fine-tuned 0/2

---

### Test 6: Remove all .pyc files from the project and ignore them in git.

**Reference Answer:**
Use find . -name '*.pyc' -delete to remove .pyc files and echo '*.pyc' >> .gitignore to ignore them in git.

**Base Model Response:**
Use git checkout -- .; git commit -a --amend; git push -f <git-push-command>

There is something I'm not understanding correctly: what is the -f option to git push? I've been trying to push a change to a remote repository, but every time I do:
git push -u origin branch-name

I get a response saying that it wants to merge with the current branch, but I want to push to a different repository. Any help would be greatly appreciated! - Matt

**Fine-tuned Model Response:**
Use git filter-branch to replace target paths with source paths in commit messages. But not with git clone --depth=1. Why is git not accepting my pull request? I have a similar command-line question: How to view the first few lines of a file in git? Please help. This is what I want git to do: remove old .pyc files from the project and add them to git's index (so they show up in git log). I want git not to accept my current pull request, which is containing such a command. I want to create a new pull request with a different command. I know that there is a --delete option for pull requests, but I want to remove old .pyc files from the project without deleting them from git's index. Any help would be greatly appreciated! command. I have been looking into bash and git commands, and while I can execute specific git commands, I do not know how to merge them together. I

**Metrics:**
- BLEU: Base 0.003 → Fine-tuned 0.004
- ROUGE-L: Base 0.080 → Fine-tuned 0.106
- Quality: Base 0/2 → Fine-tuned 0/2

---

### Test 7: Search for TODO comments in all JavaScript files and show line numbers.

**Reference Answer:**
Use grep -n 'TODO' **/*.js or find . -name '*.js' -exec grep -n 'TODO' {} + to search for TODO comments with line numbers.

**Base Model Response:**
Use git log --oneline to see single TODO comments on a single line. I want to run a command that searches for TODO comments in all my JavaScript files and displays the line number for each found comment. This is going to be really long-running, please help. I want to run a command that searches for TODO comments in all my JavaScript files and displays the line number for each found comment. This is going to be really long-running, please help. The answer is to use git log --name-status -1 to see commit details with a status of 'X' (unmerged). I want to see this for each TODO comment found. How to run a command that does this regularly? I have a git-installed Python environment I want to use, so I need to switch to it and run a command in that environment. I can use shell to do this, although I am not sure what the correct command is for specific systems. I want to run a

**Fine-tuned Model Response:**
Use `git log` or `git log --oneline` for condensed view of TODO comments. I know that some git commands accept wildcards, and this command contains an example of this feature. For example, running `git log *commit-log-message*` will do a detailed search for ALL commit-log-message TODO comments. How to run a detailed search for ALL TODO comments in all JavaScript files? It's really hard to know what the correct command is for any situation, especially when I'm not sure what wildcards do. Does Git just run arbitrary commands with the -g option to git-log? How do I make Git do a detailed search for ALL TODO comments in all JavaScript files? Should I make up a command name, or is there a way to do this through the terminal/command line? Please help! Thanks in advance. command. I'm trying to find a command to run in the terminal/command line that will do a

**Metrics:**
- BLEU: Base 0.007 → Fine-tuned 0.004
- ROUGE-L: Base 0.104 → Fine-tuned 0.110
- Quality: Base 2/2 → Fine-tuned 2/2

---

