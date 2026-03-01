# Building x265 with Multiview Support

## Why Build x265 with Multiview?

Your VR180 Silver Bullet app now supports **dual-mode MV-HEVC encoding**:

1. **Default Mode**: `spatial make` (Apple VideoToolbox)
   - ✅ Works out of the box
   - ✅ Fast (hardware accelerated)
   - ❌ Limited to Apple's encoder quality

2. **Advanced Mode**: `x265 multiview` + `spatial metadata`
   - ✅ Industry-standard x265 encoder (highest quality)
   - ✅ Full control over encoding parameters
   - ✅ Better compression efficiency
   - ❌ Requires building x265 from source

## Prerequisites

- macOS or Linux
- CMake (`brew install cmake`)
- Build tools (`xcode-select --install` on macOS)
- Git

## Build Instructions

### Step 1: Clone x265 Repository

```bash
cd ~/Downloads
git clone https://github.com/videolan/x265.git
cd x265
```

Make sure you have version 4.0 or later (released September 13, 2024+).

### Step 2: Configure Build

```bash
cd build/linux  # Or build/macOS if that exists
cmake ../../source -DENABLE_MULTIVIEW=ON
```

**Important**: The `-DENABLE_MULTIVIEW=ON` flag is critical!

### Step 3: Compile

```bash
make -j$(sysctl -n hw.ncpu)  # macOS
# or
make -j$(nproc)              # Linux
```

This will take 5-10 minutes depending on your system.

### Step 4: Install

```bash
sudo make install
```

Or install to a custom location:

```bash
sudo make install DESTDIR=/usr/local
```

### Step 5: Verify Installation

```bash
x265 --help | grep multiview-config
```

You should see:
```
--multiview-config <filename>   Multiview configuration file
```

If you see this, **success!** Your app will now automatically use x265 multiview encoding.

## Troubleshooting

### "cmake: command not found"

Install CMake:
```bash
brew install cmake
```

### "No such file or directory: build/linux"

Try:
```bash
mkdir -p build/custom
cd build/custom
cmake ../../source -DENABLE_MULTIVIEW=ON
make -j$(sysctl -n hw.ncpu)
sudo make install
```

### "Permission denied" during install

Use sudo:
```bash
sudo make install
```

Or install to your home directory:
```bash
make install PREFIX=$HOME/.local
```

Then add to PATH in `~/.zshrc` or `~/.bash_profile`:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Verify it's the correct x265

```bash
which x265
x265 --version
x265 --help | grep multiview
```

## Using the App with x265 Multiview

Once x265 with multiview support is installed:

1. **Open VR180 Silver Bullet**
2. **Select Vision Pro MV-HEVC mode**
3. **Process your video**

The app will **automatically detect** x265 multiview and use it!

You'll see this message:
```
Using x265 multiview encoding (highest quality)...
```

## Quality Comparison

### spatial make (Default)
- Uses Apple VideoToolbox
- Good quality
- Fast encoding
- ~0.5-1x realtime

### x265 multiview (Advanced)
- Uses x265 open-source encoder
- Excellent quality (better compression)
- Slower encoding
- ~0.1-0.3x realtime (depends on CPU)
- More control over encoding parameters

## File Size Comparison

For the same perceived quality:
- **spatial make**: ~350 Mbps
- **x265 multiview**: ~250-300 Mbps (20-30% smaller)

Or for the same bitrate:
- **x265 multiview** will have noticeably better quality

## Advanced x265 Parameters

If you want to customize x265 settings, edit `vr180_gui.py` line ~699:

```python
x265_cmd = [
    'x265',
    '--multiview-config', str(config_file),
    '--fps', str(fps),
    '--input-res', f'{width}x{height}',
    '--bitrate', str(bitrate_kbps),
    '--profile', 'main10',
    '--preset', 'slow',  # Add this for better compression
    '--crf', '18',       # Or use CRF instead of bitrate
    '--colorprim', 'bt709',
    '--transfer', 'bt709',
    '--colormatrix', 'bt709',
    '--output', str(hevc_file)
]
```

## Uninstalling

If you want to go back to Homebrew x265:

```bash
# Remove custom-built x265
sudo rm /usr/local/bin/x265

# Reinstall Homebrew version
brew reinstall x265
```

The app will automatically fall back to `spatial make` mode.

## Performance Tips

### Speed vs Quality

x265 preset options (fastest to slowest):
- `--preset ultrafast` - Very fast, lower quality
- `--preset fast` - Fast, good quality
- `--preset medium` - Default balance
- `--preset slow` - Slow, excellent quality (recommended)
- `--preset veryslow` - Very slow, best quality

### Disk Space

x265 multiview workflow requires temporary disk space:
- Raw YUV: ~40-80 GB per minute (4K)
- All files are auto-deleted after encoding

Make sure you have enough free space!

## Credits

- **x265 Project**: https://github.com/videolan/x265
- **Multiview Extension**: Added in x265 4.0 (2024)
- **Guide Reference**: https://spatialgen.com/blog/encode-mvhevc-with-ffmpeg/

## Version History

- **v1.2.0**: Added x265 multiview support with automatic detection ⭐ YOU ARE HERE
