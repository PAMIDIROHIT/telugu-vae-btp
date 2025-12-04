#!/bin/bash
# Organize and push commits with neat messages

cd /home/mohanganesh/ROHIT/BTP/vae_project

echo "========================================="
echo "Telugu VAE BTP - GitHub Push Helper"
echo "========================================="
echo ""

# Check if there are any uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  You have uncommitted changes:"
    git status --short
    echo ""
    echo "Would you like to commit these first? (y/n)"
    read -r response
    
    if [ "$response" = "y" ]; then
        echo ""
        echo "Select commit type:"
        echo "1) feat    - New feature"
        echo "2) fix     - Bug fix"
        echo "3) docs    - Documentation"
        echo "4) data    - Dataset changes"
        echo "5) refactor- Code refactoring"
        echo "6) test    - Tests"
        echo "7) chore   - Maintenance"
        read -p "Enter number (1-7): " commit_type
        
        case $commit_type in
            1) type="feat" ;;
            2) type="fix" ;;
            3) type="docs" ;;
            4) type="data" ;;
            5) type="refactor" ;;
            6) type="test" ;;
            7) type="chore" ;;
            *) type="chore" ;;
        esac
        
        echo ""
        read -p "Enter commit message: " message
        
        git add .
        git commit -m "$type: $message"
        echo "✅ Changes committed!"
    fi
else
    echo "✅ No uncommitted changes. Working tree is clean."
fi

echo ""
echo "========================================="
echo "Current commits (latest 5):"
echo "========================================="
git log --oneline -5
echo ""

echo "========================================="
echo "Ready to push to GitHub?"
echo "========================================="
echo ""
echo "⚠️  IMPORTANT: Make sure you've created the repository on GitHub first!"
echo ""
echo "Repository: https://github.com/PAMIDIROHIT/telugu-vae-btp"
echo ""
echo "Create it at: https://github.com/new"
echo "  - Name: telugu-vae-btp"
echo "  - DO NOT initialize with README"
echo ""
read -p "Have you created the repository? (y/n): " created

if [ "$created" = "y" ]; then
    echo ""
    echo "Pushing to GitHub..."
    git push -u origin master
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "========================================="
        echo "✅ SUCCESS! Code pushed to GitHub!"
        echo "========================================="
        echo ""
        echo "View your repository at:"
        echo "https://github.com/PAMIDIROHIT/telugu-vae-btp"
    else
        echo ""
        echo "========================================="
        echo "❌ Push failed!"
        echo "========================================="
        echo ""
        echo "Common issues:"
        echo "1. Repository doesn't exist yet - create it at https://github.com/new"
        echo "2. Authentication failed - you may need:"
        echo "   - Personal Access Token: https://github.com/settings/tokens"
        echo "   - Or SSH key: https://github.com/settings/keys"
        echo ""
        echo "For help, see: /home/mohanganesh/.gemini/antigravity/brain/*/setup_instructions.md"
    fi
else
    echo ""
    echo "Please create the repository first, then rerun this script."
fi
