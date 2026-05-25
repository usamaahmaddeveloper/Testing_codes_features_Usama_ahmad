from pytube import YouTube

url = "https://www.youtube.com/watch?v=yyUHQIec83I"  # replace with your video link
yt = YouTube(url)
# This gives a dictionary of all captions
available_captions = yt.captions

print("Available captions:")
for code, caption in available_captions.items():
    print(f"{code}: {caption.name}")