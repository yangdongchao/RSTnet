#!/bin/bash

# Initialize variables
suffix=""
# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    -il) il="$2"; shift ;;
    -s) s="$2"; shift ;;
    --suffix) suffix="$2"; shift ;;
    --device) device="$2"; shift ;;
    --ngpus) ngpus="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; usage ;;
  esac
  shift
done

device="cuda:"$((($device-1) % $ngpus))
echo "Device: $device"

audiosr -il $il -s $s --suffix "$suffix" --device $device