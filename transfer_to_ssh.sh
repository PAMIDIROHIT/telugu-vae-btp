#!/bin/bash
# Transfer Telugu VAE project folder to remote SSH server

echo "========================================="
echo "SSH File Transfer Helper"
echo "Telugu VAE BTP Project"
echo "========================================="
echo ""

# Source directory
SOURCE_DIR="/home/mohanganesh/ROHIT/BTP/vae_project"

echo "Source directory: $SOURCE_DIR"
echo ""

# Get remote server details
read -p "Enter remote username (e.g., rohit): " REMOTE_USER
read -p "Enter remote hostname/IP (e.g., 192.168.1.100): " REMOTE_HOST
read -p "Enter remote path (e.g., /home/rohit/BTP/): " REMOTE_PATH

echo ""
echo "========================================="
echo "Transfer Configuration:"
echo "========================================="
echo "From: $SOURCE_DIR"
echo "To:   $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
echo ""

# Test SSH connection first
echo "Testing SSH connection..."
ssh -o ConnectTimeout=5 "$REMOTE_USER@$REMOTE_HOST" "echo 'Connection successful!'" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ SSH connection successful!"
else
    echo "❌ SSH connection failed!"
    echo ""
    echo "Please check:"
    echo "1. Hostname/IP is correct"
    echo "2. Username is correct"
    echo "3. You have SSH access to the server"
    echo "4. SSH key is set up (run: ssh-copy-id $REMOTE_USER@$REMOTE_HOST)"
    exit 1
fi

echo ""
echo "========================================="
echo "What would you like to transfer?"
echo "========================================="
echo "1) Entire project (excluding large files)"
echo "2) Only OCROPUS dataset files"
echo "3) Only source code (no data, checkpoints)"
echo "4) Custom (you choose files)"
read -p "Enter choice (1-4): " TRANSFER_TYPE

echo ""

case $TRANSFER_TYPE in
    1)
        echo "Transferring entire project..."
        rsync -avz --progress \
            --exclude '.git/' \
            --exclude '__pycache__/' \
            --exclude '*.pyc' \
            --exclude 'checkpoints/*.pth' \
            --exclude 'checkpoints/*.pt' \
            --exclude 'experiments/*/checkpoints/*.pth' \
            --exclude 'data/raw/synthetic_default/' \
            "$SOURCE_DIR/" \
            "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
        ;;
    
    2)
        echo "Transferring OCROPUS dataset files..."
        # Create remote directory first
        ssh "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $REMOTE_PATH/data"
        
        # Transfer dataset files
        rsync -avz --progress \
            "$SOURCE_DIR/data/generate_ocropus_dataset.sh" \
            "$SOURCE_DIR/data/postprocess_organize_and_resize.py" \
            "$SOURCE_DIR/data/verify_dataset.sh" \
            "$SOURCE_DIR/data/split_train_test.py" \
            "$SOURCE_DIR/data/split_train_test.sh" \
            "$SOURCE_DIR/data/telugu_labels.txt" \
            "$SOURCE_DIR/data/telugu_lines.txt" \
            "$SOURCE_DIR/data/README.txt" \
            "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/data/"
        ;;
    
    3)
        echo "Transferring source code only..."
        rsync -avz --progress \
            --exclude '.git/' \
            --exclude '__pycache__/' \
            --exclude '*.pyc' \
            --exclude 'checkpoints/' \
            --exclude 'experiments/' \
            --exclude 'data/raw/' \
            --exclude 'data/fonts/' \
            --exclude 'logs/' \
            --exclude 'results/' \
            "$SOURCE_DIR/" \
            "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
        ;;
    
    4)
        echo "For custom transfer, use this command format:"
        echo ""
        echo "rsync -avz --progress \\"
        echo "    <source_path> \\"
        echo "    $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
        echo ""
        exit 0
        ;;
    
    *)
        echo "Invalid choice!"
        exit 1
        ;;
esac

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ Transfer completed successfully!"
    echo "========================================="
    echo ""
    echo "Verify on remote server:"
    echo "ssh $REMOTE_USER@$REMOTE_HOST 'ls -lh $REMOTE_PATH'"
else
    echo ""
    echo "========================================="
    echo "❌ Transfer failed!"
    echo "========================================="
fi
