import argparse 
def process_files(index_file, source_file, output_lines_file, cleaned_index_file):
    indices = dict()
    cleaned_lines = []

    # Read file1: extract indices and clean lines
    with open(index_file, "r") as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.strip().split()
            # First token is a float index
            idx = int(float(parts[0]))
            if idx not in indices:
                indices[idx] = 1
            else:
                indices[idx] += 1

            # Remove the first token and keep the rest
            cleaned_line = " ".join(parts[1:])
            cleaned_lines.append(cleaned_line + "\n")

    # Save selected lines from file2
    print(len(indices), "lines to extract from source file.")
    with open(source_file, "r") as f_in, open(output_lines_file, "w") as f_out:
        for i, line in enumerate(f_in):
            if i in indices:
                for _ in range(indices[i]):
                    f_out.write(line)

    # Save cleaned version of file1
    with open(cleaned_index_file, "w") as f:
        f.writelines(cleaned_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process KITTI keyframe files.")
    parser.add_argument("--index_file", type=str, required=True, help="Path to the index file (e.g., kitti05.txt)")
    parser.add_argument("--source_file", type=str, required=True, help="Path to the source poses file (e.g., 05.txt)")
    parser.add_argument("--output_lines_file", type=str, required=True, help="Path to save extracted lines (e.g., 05c.txt)")
    parser.add_argument("--cleaned_index_file", type=str, required=True, help="Path to save cleaned index file (e.g., kitti05c.txt)")

    args = parser.parse_args()

    process_files(
        index_file=args.index_file,
        source_file=args.source_file,
        output_lines_file=args.output_lines_file,
        cleaned_index_file=args.cleaned_index_file
    )
# Example usage
# process_files(
#     index_file="/home/dominic/Documents/vggt/logs/kitti05.txt",
#     source_file="/home/dominic/Downloads/data_odometry_poses/dataset/poses/05.txt",
#     output_lines_file="/home/dominic/Downloads/data_odometry_poses/dataset/poses/05c.txt",
#     cleaned_index_file="/home/dominic/Documents/vggt/logs/kitti05c.txt"
# )
