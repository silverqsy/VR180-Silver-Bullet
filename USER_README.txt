================================================================================
VR180 PROCESSOR v1.0.0
================================================================================

INSTALLATION:
────────────
Simply extract the ZIP file and double-click "VR180 Processor.app"

✅ NO additional software required!
✅ NO Python installation needed!
✅ NO FFmpeg installation needed!
✅ Everything is bundled and ready to use!

SYSTEM REQUIREMENTS:
───────────────────
• macOS 10.13 (High Sierra) or later
• 200MB free disk space

USAGE:
─────
1. Launch the application
2. Click "Browse..." to select your VR180 video
3. Adjust the settings as needed:
   - Global Horizontal Shift: Fix split-eye frames
   - Global Panomap Adjustment: Correct overall orientation
   - Stereo Offset: Fine-tune each eye individually
4. Use the preview modes to verify your adjustments
5. Click "Process Video" to create the corrected output

PREVIEW MODES:
─────────────
• Side by Side: Standard SBS output
• Anaglyph (Red/Cyan): 3D view (use red/cyan glasses)
• Overlay 50%: Semi-transparent comparison
• Difference: Highlight misalignment
• Checkerboard: Edge alignment check
• Left/Right Only: View individual eyes

ZOOM & PAN:
──────────
• Use "In" and "Out" buttons to zoom
• When zoomed in, click and drag to pan around
• Mouse wheel also zooms
• "Reset" button returns to 100%

OUTPUT OPTIONS:
──────────────
• Codec: Auto (matches input), H.265, or ProRes
• Quality: CRF 0-51 for H.265 (18 = visually lossless)
• ProRes Profile: Choose based on your needs

TROUBLESHOOTING:
───────────────
If the app says "cannot be opened":
  1. Right-click the app
  2. Select "Open"
  3. Click "Open" in the security dialog
  OR run this in Terminal:
  xattr -cr "/path/to/VR180 Processor.app"

If video loading is slow:
  • Larger videos take longer to extract preview frames
  • This is normal and expected

SUPPORT:
───────
For issues, questions, or feature requests:
  • GitHub: [Your GitHub URL]
  • Email: [Your Email]

LICENSE:
───────
MIT License - Free to use, modify, and distribute

CREDITS:
───────
Built with:
  • Python
  • PyQt6
  • FFmpeg
  • NumPy

================================================================================
Enjoy processing your VR180 videos! 🎥
================================================================================
