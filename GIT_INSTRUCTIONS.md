# Git Instructions for New Projects

## 📋 Complete Git Workflow for New Projects

### Step 1: Create Repository on GitHub

1. Go to https://github.com/new
2. Enter repository name (e.g., `my-project`)
3. Add description
4. Choose **Public** or **Private**
5. **DO NOT** initialize with README, .gitignore, or license (we'll do it locally)
6. Click "Create repository"
7. Copy the repository URL

---

## 🚀 Setup New Project Locally

### Option A: Starting Fresh (Recommended)

```bash
# Navigate to your projects folder
cd C:\Users\YourUsername\Documents\Projects

# Create new project folder
mkdir my-project
cd my-project

# Initialize git
git init

# Configure git (first time only)
git config user.name "YourGitHubUsername"
git config user.email "your-email@example.com"

# Create initial files
# (Add your project files here)

# Create .gitignore
# (Copy from template below)

# Create README.md
# (Add project description)

# Add all files
git add .

# Check what will be committed
git status

# First commit
git commit -m "Initial commit: Project setup"

# Add remote repository
git remote add origin https://github.com/YourUsername/my-project.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

### Option B: Clone Existing Repository

```bash
# Clone the repository
git clone https://github.com/YourUsername/my-project.git

# Navigate into it
cd my-project

# Configure git (if not already configured globally)
git config user.name "YourGitHubUsername"
git config user.email "your-email@example.com"

# Start working on files
```

---

## 📝 .gitignore Template

Create a `.gitignore` file in your project root:

```
# Environment variables
.env
.env.local
.env.*.local

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/
graph_venv/

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
desktop.ini

# Logs
*.log

# Test files
test_*.py
*_test.py
```

---

## 📂 Project Structure Template

```
my-project/
├── .gitignore              # Files to ignore
├── .env.example            # Template for environment variables
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── main.py                 # Main application file
├── src/                    # Source code
│   ├── __init__.py
│   ├── module1.py
│   └── module2.py
├── tests/                  # Test files
│   ├── test_module1.py
│   └── test_module2.py
├── docs/                   # Documentation
│   └── setup.md
└── data/                   # Data files (if needed)
```

---

## 💾 Daily Git Workflow

### Making Changes

```bash
# Check status
git status

# Add specific files
git add filename.py

# Or add all changes
git add .

# Commit with message
git commit -m "Add feature: description of changes"

# Push to GitHub
git push origin main
```

### Pulling Latest Changes

```bash
# Pull latest changes from remote
git pull origin main

# Or fetch without merging
git fetch origin
```

---

## 🔄 Common Git Commands

### Checking History

```bash
# View commit history
git log

# View last 5 commits
git log -5

# View commits with changes
git log -p

# View one-line summary
git log --oneline
```

### Undoing Changes

```bash
# Undo changes in working directory
git checkout -- filename.py

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# View what was undone
git reflog
```

### Branching

```bash
# Create new branch
git branch feature-name

# Switch to branch
git checkout feature-name

# Or create and switch in one command
git checkout -b feature-name

# List all branches
git branch -a

# Delete branch
git branch -d feature-name

# Push branch to GitHub
git push origin feature-name
```

### Merging

```bash
# Switch to main branch
git checkout main

# Merge feature branch
git merge feature-name

# Push merged changes
git push origin main
```

---

## 🔐 Authentication Setup (One Time)

### Using Personal Access Token (Recommended)

```bash
# Generate token at: https://github.com/settings/tokens
# Select: repo, workflow scopes

# Save credentials locally (Windows)
git config --global credential.helper wincred

# Save credentials locally (Mac)
git config --global credential.helper osxkeychain

# Save credentials locally (Linux)
git config --global credential.helper store
```

### Using SSH (Alternative)

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your-email@example.com"

# Add to GitHub: https://github.com/settings/keys

# Test connection
ssh -T git@github.com

# Use SSH URL instead of HTTPS
git remote set-url origin git@github.com:YourUsername/my-project.git
```

---

## 🚨 Troubleshooting

### Authentication Failed

```bash
# Update remote URL with token
git remote set-url origin https://YourUsername:YOUR_TOKEN@github.com/YourUsername/my-project.git

# Or use SSH
git remote set-url origin git@github.com:YourUsername/my-project.git
```

### Unrelated Histories Error

```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

### Force Push (Use Carefully!)

```bash
# Only if you know what you're doing
git push -u origin main --force
```

### Merge Conflicts

```bash
# View conflicts
git status

# Edit conflicted files manually

# After resolving
git add .
git commit -m "Resolve merge conflicts"
git push origin main
```

---

## 📋 Commit Message Best Practices

```bash
# Good commit messages
git commit -m "Add user authentication feature"
git commit -m "Fix bug in data validation"
git commit -m "Update documentation"
git commit -m "Refactor database queries"

# Bad commit messages
git commit -m "fix"
git commit -m "update"
git commit -m "changes"
```

---

## 🎯 Quick Reference Cheat Sheet

```bash
# Initial setup
git init
git config user.name "Name"
git config user.email "email@example.com"
git remote add origin <URL>

# Daily workflow
git status
git add .
git commit -m "message"
git push origin main
git pull origin main

# Branching
git checkout -b feature-name
git checkout main
git merge feature-name
git branch -d feature-name

# Viewing history
git log --oneline
git diff
git show <commit-hash>

# Undoing
git reset --soft HEAD~1
git revert <commit-hash>
git checkout -- filename
```

---

## 🔗 Useful Links

- GitHub Docs: https://docs.github.com
- Git Documentation: https://git-scm.com/doc
- GitHub CLI: https://cli.github.com
- Commit Message Guide: https://www.conventionalcommits.org

---

## ✅ Pre-Push Checklist

Before pushing to GitHub:

- [ ] All files added with `git add .`
- [ ] Commit message is clear and descriptive
- [ ] `.env` file is in `.gitignore`
- [ ] No sensitive data in code
- [ ] Tests pass (if applicable)
- [ ] README is updated
- [ ] No large files (>100MB)
- [ ] No unnecessary files committed

---

**Remember:** Always pull before pushing to avoid conflicts!
