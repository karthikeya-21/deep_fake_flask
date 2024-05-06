import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from datetime import datetime, timedelta

cred = credentials.Certificate("deep-fake-video-detection-firebase-adminsdk-fhwy5-6521f5750b.json")
firebase_admin.initialize_app(cred,{
    'storageBucket': 'deep-fake-video-detection.appspot.com'
})

bucket = storage.bucket()

def upload_video_to_storage(video_file, destination_blob_name):
    try:
        # Create a blob object representing the video file
        blob = bucket.blob(destination_blob_name)



        # Upload the video file to Firebase Storage
        blob.upload_from_file(video_file)


        expiration_time = datetime.utcnow() + timedelta(hours=1)
        # Get the public URL of the uploaded video
        video_url = blob.generate_signed_url(expiration=expiration_time)

        print(f"Video uploaded successfully. URL: {video_url}")
        return video_url

    except Exception as e:
        print(f"Error uploading video: {e}")

# # Example usage
# video_path = 'uploads/906.mp4'
# destination_blob_name = 'videos/906.mp4'  # Path in Firebase Storage
# video_url = upload_video_to_storage(video_path, destination_blob_name)

