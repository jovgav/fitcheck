# Fix GitHub Pages 404 Error

## ✅ Quick Fix Steps:

### 1. Check Your Repository Structure
Your GitHub repository should look like this:
```
fitcheck/
├── index.html          ← Main file (at root level)
├── js/
│   └── app.js
├── sample_images/
├── README.md
└── other files...
```

### 2. Enable GitHub Pages
1. Go to your GitHub repository
2. Click **Settings** tab
3. Scroll to **Pages** section
4. Set:
   - **Source**: Deploy from a branch
   - **Branch**: main (or master)
   - **Folder**: / (root)
5. Click **Save**

### 3. Wait for Build
- GitHub Pages takes 5-10 minutes to build
- You'll see a green checkmark when ready
- Check the **Actions** tab for build status

### 4. Check Your URL
- Correct: `https://yourusername.github.io/fitcheck`
- Wrong: `https://yourusername.github.io/fitcheck/static_app/`

## 🔧 Common Issues:

### Issue: "404 Not Found"
**Solution**: Make sure `index.html` is at the repository root, not in a subfolder

### Issue: "Page not found"
**Solution**: 
1. Check GitHub Pages is enabled in Settings
2. Wait 5-10 minutes for build to complete
3. Try incognito/private browsing mode

### Issue: "Repository not found"
**Solution**: Check your repository name matches the URL exactly

## 📝 Test Your Setup:

1. **Local Test**: Open `index.html` in your browser
2. **GitHub Test**: Visit `https://yourusername.github.io/fitcheck`
3. **Debug**: Check browser console for JavaScript errors

## 🆘 Still Having Issues?

1. **Check Actions tab** in your GitHub repository for build errors
2. **Verify file names** are exactly: `index.html` and `js/app.js`
3. **Clear browser cache** and try again
4. **Wait longer** - sometimes takes 15+ minutes

Your app should work once GitHub Pages builds successfully! 🚀
