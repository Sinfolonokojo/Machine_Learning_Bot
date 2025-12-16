# Fix PyTorch DLL Error on Windows

## The Problem

You're getting this error:
```
OSError: Error loading "c10.dll" or one of its dependencies
```

This happens because PyTorch needs **Microsoft Visual C++ Redistributable** to run on Windows.

---

## Quick Fix (2 minutes)

### Option 1: Download & Install Visual C++ Redistributable

**Download here:**
👉 https://aka.ms/vs/17/release/vc_redist.x64.exe

**Steps:**
1. Click the link above
2. Download `vc_redist.x64.exe`
3. Run the installer
4. Click "Install" (no configuration needed)
5. Restart your terminal
6. Run training again:
   ```bash
   venv\Scripts\activate
   python src/training/train_ppo.py
   ```

---

### Option 2: Use Python 3.11 or 3.12 Instead (Recommended)

Python 3.14 is very new and has compatibility issues. **Python 3.11 is more stable.**

**Steps:**
1. Download Python 3.11: https://www.python.org/downloads/release/python-31110/
2. Install it
3. Create new virtual environment:
   ```bash
   python3.11 -m venv venv_py311
   venv_py311\Scripts\activate
   pip install -r requirements.txt
   ```
4. Run training:
   ```bash
   python src/training/train_ppo.py
   ```

---

## Which Option Should I Choose?

- **Option 1 (Install VC++ Redistributable)**: ✅ Faster, keeps your current Python
- **Option 2 (Use Python 3.11)**: ✅ More stable, better package compatibility

**My Recommendation**: Try Option 1 first. If it doesn't work, use Option 2.

---

## After Fixing

Test that everything works:
```bash
venv\Scripts\activate
python test_install.py
```

If you see all `[OK]` messages, you're ready to train!

```bash
python src/training/train_ppo.py
```
