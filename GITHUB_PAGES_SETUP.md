# GitHub Pages Configuration

## Repository Settings
1. Go to your GitHub repository
2. Click **Settings** tab
3. Scroll down to **Pages** section
4. Configure:
   - **Source**: Deploy from a branch
   - **Branch**: main (or master)
   - **Folder**: / (root)
5. Click **Save**

## File Structure for GitHub Pages
Your repository should have this structure:
```
fitcheck/
├── index.html          # Main app (served at root)
├── js/
│   └── app.js          # JavaScript file
├── sample_images/      # Sample images
├── README.md           # Documentation
└── other files...
```

## Troubleshooting 404 Errors

### Common Issues:
1. **Wrong file location**: `index.html` must be at repository root
2. **GitHub Pages not enabled**: Check Settings → Pages
3. **Wrong branch**: Make sure you're deploying from main/master
4. **Build in progress**: Wait 5-10 minutes after enabling Pages

### Check Your URL:
- Should be: `https://yourusername.github.io/fitcheck`
- Not: `https://yourusername.github.io/fitcheck/static_app/`

### Force Refresh:
- Clear browser cache
- Try incognito/private browsing mode
- Wait 5-10 minutes for GitHub Pages to build
