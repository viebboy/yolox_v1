
#!/bin/bash

# Set the provided values for runtime, iteration, print_frequency, and src
runtime=DSP
iteration=100
print_frequency=10
src=/data/snpe_models_subset

# Create an output directory to store the result text files
output_dir="snpe_benchmark_outputs"
mkdir -p "${output_dir}"

# Count the total number of .dlc files and initialize the processed count
total_files=$(find "${src}" -maxdepth 1 -name "*.dlc" | wc -l)
processed_files=0

# Record the start time
start_time=$(date +%s)

# Iterate through all .dlc files in the src directory
for input_file in "${src}"/*.dlc; do
  # Create the output text file with the same base name as the input .dlc file
  base_name=$(basename "${input_file}" .dlc)
  output_file="${output_dir}/${base_name}.txt"

  # Check if the output text file already exists
  if [ ! -f "${output_file}" ]; then
    # Run cmd1, filter the output lines and extract the milliseconds

    milliseconds=$(enigma-ml --model "${input_file}" --runtime "${runtime}" -n "${iteration}" --timer="${print_frequency}" | awk '/Total/ {print $(NF-1)}' | sed 's/P//')
    echo ${milliseconds}
    # Calculate the average of the extracted milliseconds
    count=0
    total=0
    for ms in ${milliseconds}; do
      count=$((count + 1))
      total=$(echo "${total} ${ms}" | awk '{ printf "%.2f", $1 + $2 }')
    done
    if [ "${count}" -eq 0 ]; then
      echo "Error: No matching lines found for ${input_file}. Skipping this file."
    else
      average=$(echo "${total} ${count}" | awk '{ printf "%.2f", $1 / $2 }')

      # Write the average to the output text file
      echo "average: ${average}"
      echo "${average}" > "${output_file}"
    fi

    # Write the average to the output text file
    echo "${average}" > "${output_file}"
  fi

  # Increment the processed files count
  processed_files=$((processed_files + 1))

  # Calculate the percentage of completion
  percentage=$(echo "${processed_files} ${total_files}" | awk '{ printf "%.2f", ($1 / $2) * 100 }')

  # Calculate the elapsed time
  current_time=$(date +%s)
  elapsed_time=$((current_time - start_time))

  # Print the percentage of completion and elapsed time
  echo "Progress: ${percentage}% (${processed_files}/${total_files}), Elapsed Time: ${elapsed_time} seconds"
done

