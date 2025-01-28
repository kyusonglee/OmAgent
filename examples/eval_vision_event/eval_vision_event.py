import os
os.environ["OMAGENT_MODE"] = "lite"

# Import required modules and components
from omagent_core.utils.container import container
from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.advanced_components.workflow.vision_event.workflow import VisionEventWorkflow
from pathlib import Path
from omagent_core.utils.registry import registry
from omagent_core.clients.devices.programmatic import ProgrammaticClient
from omagent_core.utils.logger import logging
import argparse
import json
import os
from glob import glob


def parse_args():
    """Parse command line arguments for vision event detection evaluation"""
    parser = argparse.ArgumentParser(description='Evaluate vision event detection system')
    parser.add_argument('--endpoint', type=str, default="https://api.openai.com/v1",
                        help='OpenAI API endpoint')
    parser.add_argument('--api_key', type=str, default=None,
                        help='OpenAI API key')
    parser.add_argument('--model_id', type=str, default="openbmb/MiniCPM-o-2_6",
                        help='Model ID to use (default: MiniCPM)')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Directory containing images to analyze')
    parser.add_argument('--event_prompt', type=str, required=True,
                        help='Event to detect (e.g., "detect if a person is female")')
    parser.add_argument('--output_path', type=str, default='output',
                        help='Path to output directory')
    parser.add_argument('--image_extensions', type=str, default='.jpg,.jpeg,.png',
                        help='Comma-separated list of image extensions to process')
    return parser.parse_args()

def get_image_files(directory, extensions):
    """Get list of image files from directory with specified extensions"""
    extensions = extensions.split(',')
    image_files = []
    for ext in extensions:
        image_files.extend(glob(os.path.join(directory, f'*{ext}')))
    return sorted(image_files)

def main():
    """Main function to run vision event detection evaluation"""
    # Parse command line arguments
    args = parse_args()

    # Set environment variables for OpenAI API
    #os.environ["custom_openai_endpoint"] = args.endpoint
    #os.environ["custom_openai_key"] = args.api_key
    os.environ["model_id"] = args.model_id

    # Get list of image files to process
    image_files = get_image_files(args.images_dir, args.image_extensions)
    if not image_files:
        #logging.error(f"No images found in {args.images_dir} with extensions {args.image_extensions}")
        return

    # Setup logging and paths
    logging.init_logger("omagent", "omagent", level="INFO")
    CURRENT_PATH = Path(__file__).parents[0]

    # Initialize agent modules and configuration
    registry.import_module(project_path=CURRENT_PATH.joinpath('agent'))
    container.from_config(CURRENT_PATH.joinpath('container.yaml'))

    # Setup Vision Event workflow
    workflow = ConductorWorkflow(name='vision_event')
    vision_workflow = VisionEventWorkflow()
    vision_workflow.set_input(image_path=workflow.input('image_path'), 
                            event_prompt=workflow.input('event_prompt'))
    workflow >> vision_workflow
    workflow.register(overwrite=True)

    # Initialize programmatic client
    config_path = CURRENT_PATH.joinpath('configs')
    programmatic_client = ProgrammaticClient(processor=workflow, config_path=config_path)

    # Prepare batch processing inputs
    output_json = []
    workflow_input_list = []
    
    for image_path in image_files:
        workflow_input_list.append({
            "image_path": image_path,
            "event_prompt": args.event_prompt
        })
    
    # Process images in batches
    res = programmatic_client.start_batch_processor(workflow_input_list=workflow_input_list, max_tasks=5)
    
    # Collect results
    for r, w in zip(res, workflow_input_list):
        #print (w,r)
        output_json.append({
            "image_path": w['image_path'],
            "event_prompt": w['event_prompt'],
            "detection_result": r,
        })
        
    # Prepare final output
    final_output = {
        "model_id": args.model_id,
        "event_prompt": args.event_prompt,
        "total_images": len(image_files),
        "detection_results": output_json
    }

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Save results to output file
    if "positive" in args.images_dir:
        sent = "positive"
    else:
        sent = "negative"

    output_filename = f'{sent}_{args.model_id.replace("/", "_")}.json'
    output_path = os.path.join(args.output_path, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=4)

    # Log summary statistics
    detected_count = sum(1 for result in output_json if result['detection_result'])
    logging.info(f"Processed {len(image_files)} images")
    logging.info(f"Event detected in {detected_count} images")
    logging.info(f"Results saved to {output_path}")

    # Cleanup
    programmatic_client.stop_processor()

if __name__ == "__main__":
    main() 
