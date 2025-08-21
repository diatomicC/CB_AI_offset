#!/bin/bash

# GitHub Actions Setup Script

echo "🚀 GitHub Actions CI/CD Setup Started"
echo "=================================================="

# Get GitHub repository information
read -p "📝 Enter GitHub username: " GITHUB_USERNAME
read -p "📝 Enter GitHub repository name: " GITHUB_REPO

# Create .github directory
echo "📁 Creating .github directory..."
mkdir -p .github/workflows

# Update badge URLs in README.md
echo "🔧 Updating README.md badge URLs..."
sed -i.bak "s/{username}/$GITHUB_USERNAME/g" README.md
sed -i.bak "s/{repo}/$GITHUB_REPO/g" README.md

# Clean backup files
rm -f README.md.bak

echo "✅ GitHub Actions setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Push code to GitHub:"
echo "   git add ."
echo "   git commit -m 'Add GitHub Actions workflows'"
echo "   git push origin main"
echo ""
echo "2. Check Actions tab in GitHub repository"
echo "3. Verify workflows run automatically"
echo ""
echo "4. Test manual execution:"
echo "   - Actions tab → Build Windows Executable → Run workflow"
echo ""
echo "🎯 Key features:"
echo "   - Automatic Windows executable build"
echo "   - Build in Python 3.11 environment"
echo "   - Automatic deployment package creation"
echo "   - Save results to GitHub artifacts"
echo ""
echo "📁 Generated files:"
echo "   - .github/workflows/build.yml"
echo "   - .github/workflows/build-windows.yml"
echo "   - GITHUB_ACTIONS_GUIDE.md"
echo "   - setup_github_actions.sh"
echo ""
echo "🚀 Now you can build automatically with GitHub Actions!"
