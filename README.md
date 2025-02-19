# Video Supercut

This project processes videos by extracting audio, detecting speech segments, transcribing with Whisper, removing duplicate phrases, and compiling a final edited video.

## Prerequisites

- Python 3.6 or higher
- [ffmpeg](https://ffmpeg.org/) must be installed and available in your system PATH (the script will try to use imageio-ffmpeg if you haven't read this).
- [Git](https://git-scm.com/) for cloning the repository.

## Setup

1. **Clone the repository:**
   ```bash
   git clone git@github.com:justinkahrs/video-supercut.git
   ```
2. **Navigate into the project directory:**
   ```bash
   cd video-supercut
   ```
3. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source ./venv/bin/activate
   ```
4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Script

1. **Add Video Files:**

   - `mkdir raw`
   - Place your `.mp4` video files in the `raw` folder.

2. **Run the Script:**
   ```bash
   python3 main.py
   ```
   The script will process the videos and output the edited videos to the `edited_videos` folder.

## Notes

- The first run may download the necessary Whisper model weights.
- Temporary files are stored in the system's temporary directory.
- Ensure that ffmpeg is installed and configured correctly on your system.
