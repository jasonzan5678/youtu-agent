import yt_dlp


class DownloadBilibiliVideo:
    def download_video(self, url: str, output_path: str = None) -> str:
        """Download a Bilibili video

        Args:
            url (str): The URL of the video to download.
            output_path (str, optional): The path to save the downloaded video. Defaults to None.

        Returns:
            str: Path where video was downloaded
        """
        try:
            ydl_opts = {"format": "bestvideo+bestaudio/best"}
            if output_path:
                ydl_opts["outtmpl"] = output_path
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return output_path or "Download completed"
        except Exception as e:
            return f"Download failed: {str(e)}"
