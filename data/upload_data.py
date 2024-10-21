from argparse import ArgumentParser

from minio import Minio
from minio.error import S3Error


def main(endpoint, access_key, secret_key, bucket_name, source_file, destination_file):
    client = Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
    )

    # The file to upload, change this path if needed
    source_file = source_file

    # The destination bucket and filename on the MinIO server
    bucket_name = bucket_name
    destination_file = destination_file

    # Make the bucket if it doesn't exist.
    found = client.bucket_exists(bucket_name)
    if not found:
        client.make_bucket(bucket_name)
        print("Created bucket", bucket_name)
    else:
        print("Bucket", bucket_name, "already exists")

    # Upload the file, renaming it in the process
    client.fput_object(
        bucket_name,
        destination_file,
        source_file,
    )
    print(
        source_file,
        "successfully uploaded as object",
        destination_file,
        "to bucket",
        bucket_name,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--endpoint", type=str, required=True)
    parser.add_argument("--access_key", type=str, required=True)
    parser.add_argument("--secret_key", type=str, required=True)
    parser.add_argument("--bucket_name", type=str, required=True)
    parser.add_argument("--source_file", type=str, required=True)
    parser.add_argument("--destination_file", type=str, required=True)
    args = parser.parse_args()
    try:
        main(
            args.endpoint,
            args.access_key,
            args.secret_key,
            args.bucket_name,
            args.source_file,
            args.destination_file,
        )
    except S3Error as exc:
        print("error occurred.", exc)
