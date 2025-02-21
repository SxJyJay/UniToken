import os
import sys
import glob
import json

sys.path.append(os.path.abspath(__file__).rsplit("/", 2)[0])

import argparse
import builtins
import datetime
import multiprocessing as mp
import traceback
from typing import List, Optional
import random

import gradio as gr
import torch
from PIL import Image

from data.item_processor import generate_crop_size_list
from inference_solver_base import FlexARInferenceSolverBase
from inference_solver_anyres import FlexARInferenceSolverAnyRes
from xllmx.util.misc import random_seed


def get_custom_prompt():
    captions = [
        "A serene sunset over a calm ocean with a silhouette of a sailboat",
        "A bustling city street in Tokyo at night with neon signs",
        "A futuristic cityscape with flying cars and towering skyscrapers",
        "A picturesque mountain landscape with a clear blue lake",
        "A cozy cabin in the woods during a snowy winter",
        "A vibrant street market in Marrakech with colorful textiles and spices",
        "A fantasy castle on a floating island in the sky",
        "A peaceful Zen garden with a koi pond and stone lanterns",
        "A group of astronauts exploring the surface of Mars",
        "A medieval knight in shining armor standing in a lush forest",
        "A whimsical fairy tale forest with magical creatures",
        "A detailed close-up of a blooming sunflower",
        "A majestic lion resting on a rock in the savannah",
        "A futuristic robot in a high-tech laboratory",
        "A serene beach with palm trees and a hammock",
        "A bustling street in Paris with the Eiffel Tower in the background",
        "A group of children playing in a colorful playground",
        "A traditional Japanese tea ceremony in a tatami room",
        "A close-up of a butterfly on a vibrant flower",
        "A dramatic stormy sky with lightning over a dark ocean",
        "A cozy coffee shop with people reading and chatting",
        "A beautiful ballet dancer performing on stage",
        "A rustic farmhouse surrounded by fields of lavender",
        "A futuristic space station orbiting a distant planet",
        "A serene waterfall in a lush tropical rainforest",
        "A vintage steam train traveling through a snowy landscape",
        "A majestic eagle soaring over a mountain range",
        "A group of penguins huddled together in Antarctica",
        "A detailed macro shot of a dew-covered spider web",
        "A bustling farmers market with fresh produce and flowers",
        "A serene desert landscape with sand dunes and a camel",
        "A traditional Indian wedding with colorful attire and decorations",
        "A futuristic underwater city with glowing buildings",
        "A cozy living room with a roaring fireplace and a Christmas tree",
        "A dramatic volcanic eruption with flowing lava",
        "A group of friends having a picnic in a sunny park",
        "A serene lake surrounded by autumn foliage",
        "A bustling New York City street with yellow taxis",
        "A whimsical treehouse in a large oak tree",
        "A detailed close-up of a honeybee on a flower",
        "A majestic elephant walking through the African savannah",
        "A futuristic city at night with glowing neon lights",
        "A serene meadow with wildflowers and butterflies",
        "A dramatic cliffside castle overlooking the ocean",
        "A cozy bedroom with fairy lights and a canopy bed",
        "A bustling street in Bangkok with food stalls and tuk-tuks",
        "A dramatic night sky with the Milky Way and shooting stars",
        "A peaceful river winding through a dense forest",
        "A futuristic robot companion in a modern home",
        "A group of dolphins jumping out of the ocean",
        "A serene mountain village with traditional wooden houses",
        "A bustling street in Rio de Janeiro during Carnival",
        "A dramatic ice cave with blue glowing ice formations",
        "A serene garden with cherry blossom trees in full bloom",
        "A futuristic spaceship flying through a nebula",
        "A cozy library with tall bookshelves and a reading nook",
        "A dramatic sunset over a desert landscape with cacti",
        "A bustling street in Mumbai with colorful markets and rickshaws",
        "A serene lake with a wooden dock and a rowboat",
        "A futuristic city with green spaces and sustainable buildings",
        "A whimsical candy land with giant lollipops and gumdrop trees",
        "A serene beach at sunrise with gentle waves",
        "A bustling street in Istanbul with historic buildings and bazaars",
        "A dramatic thunderstorm over a wheat field",
        "A cozy kitchen with a baking scene and fresh cookies",
        "A futuristic drone delivering a package to a modern home",
        "A serene forest path with sunlight filtering through the trees",
        "A bustling street in London with red double-decker buses",
        "A dramatic mountain peak covered in snow",
        "A cozy reading corner with a comfortable chair and a lamp",
        "A bustling street in Hong Kong with towering skyscrapers",
        "A serene pond with lily pads and a small wooden bridge",
        "A futuristic city with flying taxis and advanced technology",
        "A whimsical underwater scene with colorful fish and coral",
        "A serene beach with seashells and gentle waves",
        "A bustling street in Rome with historic architecture",
        "A dramatic sunset over a field of sunflowers",
        "A cozy café with people enjoying coffee and pastries",
        "A futuristic city with a monorail and green rooftops",
        "A serene meadow with grazing deer and wildflowers",
        "A bustling street in Beijing with traditional lanterns",
        "A dramatic waterfall cascading into a rocky pool",
        "A cozy living room with a large window and a view of the mountains",
        "A futuristic robot assistant in a high-tech kitchen",
        "A serene lake at dusk with reflections of the sky",
        "A bustling street in Cairo with historic mosques and markets",
        "A dramatic sunset over a rocky coastline",
        "A cozy bedroom with a quilted bedspread and soft lighting",
        "A bustling street in Sydney with the Opera House in the background",
        "A serene garden with a stone pathway and blooming flowers",
        "A futuristic city with holographic advertisements and advanced infrastructure",
        "A whimsical fairy tale castle with turrets and a drawbridge",
        "A serene forest clearing with a small wooden cabin",
        "A bustling street in San Francisco with the Golden Gate Bridge",
        "A dramatic stormy sky over a rolling countryside",
        "A cozy attic room with a skylight and vintage furniture",
        "A bustling street in Mexico City with colorful buildings and murals",
        "A serene riverbank with wildflowers and a wooden bench",
        "A futuristic city with autonomous vehicles and smart technology",
        "A whimsical enchanted forest with glowing mushrooms and fairies",
        "A serene beach with driftwood and gentle waves",
        "A bustling street in Athens with ancient ruins and modern shops",
        "A dramatic sunset over a vineyard with rows of grapevines"
    ]
    return captions


def inference_t2i(args, prompts, save_root):
    os.makedirs(save_root, exist_ok=True)
    # inference_solver = FlexARInferenceSolverBase(
    #     model_path=args.pretrained_path,
    #     precision=args.precision,
    #     target_size=args.target_size,
    # )
    inference_solver = FlexARInferenceSolverAnyRes(
        model_path=args.pretrained_path,
        precision=args.precision,
        target_size=args.target_size,
    )
    # Set some hyper-params
    # gen_t, cfg, image_top_k = 1.0, 5.5, 4000
    gen_t, cfg, image_top_k = 1.0, 3, 4000

    # Use customized prompts
    # prompts = get_custom_prompt()[:13]

    for i, prompt in enumerate(prompts):
        prefix = '_'.join(args.pretrained_path.split('/')[-2:])
        # if os.path.exists(f"{save_root}/{prefix}_{prompt}.jpg"):
        #     continue

        question = f'Generate an image according to the following prompt:\n{prompt}'
        generated = inference_solver.generate_img(
            [],
            [[question, None]],
            1536,
            gen_t,
            logits_processor=inference_solver.create_logits_processor(
                cfg=cfg, text_top_k=5, image_top_k=image_top_k
            ),
        )
        ret = {"text": generated[0], "image": generated[1], "prompt": prompt, "end_of_content": True}
        
        try:
            ret["image"][0].save(f"{save_root}/{prefix}_{i}.jpg")
        except:
            continue
        
        # ret["image"][0].save(f"{save_root}_Lumina_{prompt}.jpg")

    return

def inference_i2t(args, prompts, image_paths):
    
    save_root = args.save_root
    os.makedirs(save_root, exist_ok=True)
    # inference_solver = FlexARInferenceSolverBase(
    #     model_path=args.pretrained_path,
    #     precision=args.precision,
    #     target_size=args.target_size,
    # )
    inference_solver = FlexARInferenceSolverAnyRes(
        model_path=args.pretrained_path,
        precision=args.precision,
        target_size=args.target_size,
    )
    prompt = random.choice(prompts)

    answers = []
    for image_path in image_paths:
        image = [Image.open(image_path).convert('RGB')]
        qas = [[prompt, None]]
        generated = inference_solver.generate(
            images=image,
            qas=qas,
            max_gen_len=4096,
            temperature=1.0,
            logits_processor=inference_solver.create_logits_processor(cfg=4.0, image_top_k=2000),
        )
        answer = generated[0]
        generated_image = generated[1]  # Namely the list of newly generated images, should typically be empty in this case.
        save_format = {
            "image": image_path,
            "prompt": prompt,
            "answer": answer
        }
        answers.append(save_format)
    output_path = os.path.join(save_root, "VQA_results.json")
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(answers, outfile, ensure_ascii=False, indent=4)
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("X-LLM-X Inference Script")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--gpu_ids",
        type=int,
        nargs="+",
        help="A list of space-separated gpu ids to run the model on. "
        "The model will span across GPUs in tensor-parallel mode.",
    )
    group.add_argument(
        "--n_gpus",
        type=int,
        default=1,
        help="Number of GPUs to run the model on. Equivalent to " "--gpu_ids 0 1 2 ... n-1",
    )
    parser.add_argument("--pretrained_path", type=str, required=True, help="Path to the model checkpoints.")
    parser.add_argument(
        "--inference_mode", 
        type=str, 
        choices=["T2I", "I2T"], 
        default="T2I",
        help="T2I or I2T mode for model inference."
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "bf16"],
        default="bf16",
        help="The dtype used for model weights and inference.",
    )
    parser.add_argument(
        "--target_size", type=int, default=768, choices=[512, 768, 1024], help="The target image generation size."
    )
    parser.add_argument(
        "--save_root",
        type=str,
        help="visualization save root",
    )
    args = parser.parse_args()

    if args.inference_mode == "T2I":
        # T2I test samples
        # sentences = ["A blue T-shirt", "A boat sailing on the sea", "一个小男孩在木屋前的草地上玩耍", "一个红色枕头"]
        # sentences = [
        #     "A man up to bat during a baseball game.",
        #     "Three horses pulling a cart with a man riding it",
        #     "A yellow vase full of blooming purple tulips",
        #     "A adult and a child on the edge of some water",
        #     "The desert is on the dish ready to be eaten",
        #     "A living area with chair, sofa and television by windows."
        # ]
        # sentences = [
        #     "A bustling city street in Tokyo at night with neon signs",
        #     "A cozy cabin in the woods during a snowy winter",
        #     "A futuristic cityscape with flying cars and towering skyscrapers",
        #     "A serene sunset over a calm ocean with a silhouette of a sailboat",
        #     "A peaceful Zen garden with a koi pond and stone lanterns",
        #     "A picturesque mountain landscape with a clear blue lake",
        #     "A vibrant street market in Marrakech with colorful textiles and spices"
        # ]
        # sentences = [
        #     "A couple of men holding Wii game controllers.",
        #     "A stop sign in winter following a recent snow.",
        #     "Ten brown bears in a zoo on a snowy day.",
        #     "A person in costume and a man wait in a bus stop.",
        #     "A pole has a large clock on top next to a tall building with stairs that lead up to a doorway.",
        #     "Closeup of a wooden desk with a laptop plugged into a charger",
        #     "A busy street with cars parked down one side and motorcycles parked on the other.",
        # ]
        # sentences = [
        #     # "This image displays an ice cream cone filled with an assortment of vibrant, cartoon-style flowers. The ice cream cone itself is made of a warm, yellow color that stands out against the white background. The flowers are numerous and colorful, including yellows, pinks, blues, and purples, with various types such as daisies, sunflowers, and others. They seem to radiate out from the top of the ice cream cone, creating a sense of abundance and joy. The image has a playful, whimsical quality to it, with a clear focus on the ice cream cone and its floral contents. The style is reminiscent of watercolor painting, with soft edges and a bright, airy color palette.",
        #     # "This image captures an adorable Corgi puppy sitting on a wooden floor. The puppy is looking directly at the camera, its tongue playfully sticking out, conveying a sense of joy and anticipation. Its fur is a mix of tan and white, with pointed ears and a short tail, typical of the Corgi breed. The wooden floor provides a warm, natural backdrop that complements the puppy's colors. The photograph is taken from a slightly elevated angle, looking down at the puppy, which gives the image a playful, welcoming feel. The style of the photograph is lifelike and endearing, with a clear focus on the puppy, making it the undisputed subject of the image.\nIllustrate the description by producing an image that captures the essence and specific details mentioned in the description.",
        #     # "This image features a single, striking element: a large, black, expressive eye set against an intense red background. The eye is detailed and takes up a central position in the image, with its black color creating a strong visual contrast against the red backdrop. The background is textured, adding depth to the image. The photographic style is abstract, with a focus on color and form. The image makes no explicit use of photographic techniques such as focal length, depth of field, angle, or type of lens. The central placement and size of the eye suggest it is the primary subject of the image. There is no additional quantity or spatial relationship between multiple entities as this image contains only a single, central eye. The image does not provide sufficient information to determine if it is presented in a two- or three-dimensional space. \nCraft an image that corresponds with the caption, ensuring that the main subjects and background are accurately portrayed.",
        #     # "This image showcases an aerial view of an open hole in a construction site, possibly for a foundation. The hole is rectangular, with one shorter side and one longer side, filled with construction debris. The construction site itself is filled with various sizes of rocks. The photographic style is realistic, with an emphasis on earthy tones. The image is shot from directly above, using a wide-angle lens, which allows for the inclusion of more detail within the frame. The image does not contain any text.\nIllustrate the description by producing an image that captures the essence and specific details mentioned in the description.",
        #     # "This is a digitally created image featuring an otherworldly female character with blue skin and pointed ears, seated in a meditative lotus position. She wears an intricate headdress and is adorned with various beads and earrings. Her closed eyes and serene expression suggest introspection. The character is surrounded by rock formations and flying creatures, all set against a hazy, misty background. The image conveys a sense of fantasy and otherworldliness.",
        #     # "This image portrays a whimsical fairy tale scene, featuring a large, white-capped mushroom with a house built into it. The mushroom's cap functions as a canopy, while the stem supports a spiral staircase that leads to a second story window. Below the staircase, a door opens to a stone pathway. Two smaller mushrooms flank the stairs, and the house is surrounded by various plants and rocks. The image is hand-drawn, exhibiting a penchant for detail and a playful, imaginative style.\nVisualize the scene described in the caption and produce an image that accurately reflects its content."
        # ]

        # GenEval testing samples
        # Raw
        # sentences = [
        #     "a photo of a backpack below a cake",
        #     "a photo of a backpack right of a sandwich",
        #     "a photo of a backpack",
        #     "a photo of a banana",
        #     "a photo of a baseball bat and a bear",
        #     "a photo of a baseball bat and a giraffe",
        #     "a photo of a baseball bat"
        # ]
        # After rewrite
        # sentences = [
        #     "A sturdy, weathered leather backpack rests beneath a multi-tiered cake decorated with vibrant pastel frosting and intricate floral designs. The cake's layers tilt slightly, as if precariously balanced, with dripping icing glistening in soft ambient lighting. A rustic wooden table supports the scene, surrounded by scattered crumbs, a half-burned candle, and a notebook spilling out of the backpack. The background hints at a cozy, sunlit room with bookshelves and potted plants.",
        #     "A well-worn, olive-green backpack sits neatly to the right of a freshly made sandwich on a rustic wooden table. The sandwich, stacked high with layers of crisp lettuce, juicy tomato, and savory deli meats, rests on a white napkin. Soft sunlight filters through a nearby window, casting a warm glow across the scene. In the background, a plant’s vibrant green leaves add a touch of freshness, and a coffee cup sits beside the sandwich, steam rising gently.",
        #     "A rugged, canvas backpack rests against a weathered brick wall, its zippers slightly open to reveal the interior. The dark brown straps are worn from use, while patches and small pins decorate the exterior, showcasing personal touches. The backpack's surface bears the marks of adventure, with faded areas from sun exposure and a few scuffs from travel. In the background, a hint of a bustling urban street adds context, with sunlight streaming through the gaps between buildings, casting long shadows.",
        #     "A ripe, golden-yellow banana rests on a polished wooden countertop, its smooth skin slightly speckled with small brown spots. Soft natural light from a nearby window highlights the banana’s curves, casting a gentle shadow on the surface. The background features a few scattered fruit peels and a ceramic fruit bowl, filled with vibrant apples and oranges, adding to the fresh and inviting atmosphere of the kitchen.",
        #     "A sleek wooden baseball bat lies on the grassy forest floor, its polished surface gleaming in the dappled sunlight filtering through the trees. Nearby, a large, majestic bear stands upright on its hind legs, its fur a deep, rich brown that contrasts with the vibrant green of the underbrush. The bear's gaze is focused on the bat, curious yet calm, as if it's contemplating the scene. The forest around them is quiet, with only the distant rustle of leaves breaking the stillness.",
        #     "A worn baseball bat rests on the soft forest floor, partially covered by fallen leaves and twigs. Just a few feet away, a massive brown bear stands, its thick fur blending with the shadows of the trees. The bear's powerful paws are inches from the bat, its eyes focused on the object with a sense of quiet curiosity. Sunlight filters through the canopy, casting streaks of light across the scene, creating a surreal contrast between the wild, untamed nature and the simple, man-made bat.",
        #     "A well-used baseball bat rests on a rustic wooden bench, its smooth surface worn and etched with marks of countless games. The bat’s handle is wrapped in fraying tape, showing signs of its age and history. Sunlight filters through the nearby trees, casting a soft glow on the bat’s surface, while in the background, a baseball glove and a few scattered balls add to the nostalgic atmosphere of a quiet ballpark on a warm afternoon.",
        # ]
        # Distill from DALLE-3
        # sentences = [
        #     "A photo of a backpack placed below a cake on a table. The backpack is colorful with a zipper and straps, resting on the floor. The cake is beautifully decorated with frosting and topped with candles. The scene is set in a well-lit room, with a simple, clean background.",
        #     "A photo of a colorful backpack placed to the right of a delicious sandwich on a table. The sandwich is filled with layers of fresh vegetables, cheese, and meats, with some lettuce peeking out. The backpack is vibrant, with multiple colors and zippers. The scene is set in a cozy, well-lit room with a neutral background.",
        #     "A photo of a baseball bat placed next to a bear in a natural outdoor setting. The baseball bat is wooden with a smooth finish, lying on the ground beside the bear. The bear is standing or sitting in a calm and natural posture, surrounded by trees and greenery. The lighting is soft, and the background depicts a peaceful forest environment.",
        #     "A photo of a baseball bat placed next to a giraffe in a natural outdoor setting. The baseball bat is leaning on the ground, with the giraffe standing tall beside it. The giraffe's long neck reaches up, and the scene is set in a savannah-like environment with a clear sky, trees, and grasses. The lighting is natural, and the background is peaceful and expansive.",
        #     "A scene where an airplane is partially hidden by a large bag. The airplane is visible only from a portion, with the wings and some of the tail visible as it is obscured by the bag placed in front of it. The bag is large, colorful, and made of durable material, placed on the ground. The setting is outdoors, with a soft, natural background, and the focus is on the airplane and the bag.",
        #     "A scene where an airplane is partially hidden by a man standing in front of it. The man is tall, wearing casual clothes, and his body partially blocks the view of the airplane behind him. The airplane is mostly obscured, with only parts of its wings or tail visible. The background is an outdoor setting, with soft natural lighting and some greenery.",
        #     "A scene where an airplane is partially hidden by a turtle. The turtle is sitting on the ground, with its shell large enough to obscure the view of the airplane behind it. The airplane is only partly visible, with portions of its wings or body peeking out from behind the turtle. The setting is outdoors, with soft lighting and a natural background."
        # ]

        # For qualitative results
        # sentences = [
        #     "A serene enchanted forest at twilight, with ancient twisted trees covered in glowing blue moss. Soft rays of light filter through the dense canopy, and delicate ethereal butterflies hover near crystal-clear pools of water.",
        #     "A modern city street at night, illuminated by vibrant neon signs in Japanese kanji. In the foreground, a stoic samurai in futuristic armor with glowing katana blades stands amidst light rain, reflecting bright hues on the pavement.",
        #     "A surreal alien planet with a pinkish-orange sky and two massive moons on the horizon. Crystal-like mountain peaks shimmer in various hues, and bioluminescent plants line the shores of a shimmering purple lake.",
        #     "A bustling 1920s European train station filled with people in period clothing. Steam billows from a majestic black locomotive, while travelers with vintage luggage converse under ornate iron arches and golden clock towers.",
        #     "A breathtaking scene of floating islands suspended above the clouds, connected by hanging vines and bridges. Each island is covered in lush greenery, with cascading waterfalls that fade into mist before touching the earth below.",
        #     "A close-up portrait of a confident woman with vibrant paint splatters across her face. Her expressive eyes gaze directly at the viewer as dynamic brushstrokes blend surreal elements into her flowing hair.",
        #     "A desolate world under a blood-red sky, with the skeletal remains of skyscrapers silhouetted in the distance. Sand dunes have overtaken the landscape, and a lone wanderer wrapped in tattered gear trudges through the arid expanse.",
        #     "A powerful Bengal tiger standing on a misty forest path at dawn. Sunbeams cut through the morning fog, illuminating the tiger's vibrant orange coat and piercing green eyes as it confidently surveys its surroundings.",
        #     "A lone nomad in sleek sci-fi armor, carrying a high-tech staff, walking across vast dunes of golden sand. The background features distant shimmering towers and a sky filled with twin suns.",
        #     "A peaceful snow-covered valley illuminated by the soft glow of the northern lights. Frost-covered trees glisten under a starry sky, and a quaint wooden cabin with smoke rising from its chimney stands by a frozen river.",
        # ]
        # sentences = [
        #     "An intricately detailed brass compass with engraved markings on its circular frame. The aged metal shows subtle tarnish, and the needle points firmly north. The background is a map of ancient sea routes faded with time.",
        #     "A shimmering crystal goblet with intricate floral engravings etched into the glass. Light refracts through the facets, casting colorful rainbows across a marble surface.",
        #     "A single, vibrant red feather with fine barbs gently curved and tapered toward the tip. The quill is smooth and cream-colored, while subtle shadows highlight its delicate structure against a soft beige backdrop.",
        #     "An old-fashioned metal lantern with a weathered bronze patina. The glass panels are slightly foggy from years of use, and a faint flickering flame glows inside, casting warm light across a dark wooden surface.",
        #     "A smooth, oval-shaped jade stone with deep, translucent green tones and subtle white veins running through its surface. The stone sits elegantly on a black velvet cloth.",
        #     "A beautifully crafted silver pocket watch with ornate floral engravings on its cover. The open face reveals Roman numerals and finely crafted golden hands frozen at 10:10, resting on a lace-lined wooden surface.",
        #     "A single, flowing silk ribbon in a soft pastel pink hue, draped elegantly in loose curves. The fabric catches the light, revealing its smooth and slightly reflective texture.",
        #     "A thick white pillar candle, partially melted, inside a clear glass holder. The flame flickers gently, creating golden reflections on the glass and casting soft shadows.",
        #     "A small, pale blue egg with delicate brown speckles scattered across its smooth surface. The egg rests in a nest of dried twigs and moss.",
        #     "A rustic ceramic cup with earthy brown and beige tones, featuring a textured, slightly uneven surface. Faint glaze drips add character as it sits on a stone countertop.",
        #     "A close-up of a single monarch butterfly wing, showcasing intricate patterns of orange, black, and white. The delicate veins are highlighted by soft natural light.",
        #     "A small, transparent glass marble with swirling blue and gold patterns encased inside. It sits on a reflective surface, casting subtle light distortions.",
        #     "A beautifully aged leather-bound journal with a strap closure. The surface is textured with creases and worn edges, while faint embossing of a floral pattern adorns the cover.",
        #     "A single dark green leaf glistening with fresh raindrops. The veins are clearly visible, and droplets pool at the edges, refracting soft natural light."
        # ]
        sentences = [
            "A giant sea turtle with a vibrant coral reef on its back swimming gracefully through a crystal-clear ocean. Colorful schools of fish swirl around it, and rays of golden sunlight pierce the water’s surface.",
            "A bustling Victorian-era city with brass gears, clock towers, and steam-powered vehicles. Citizens in elaborate steampunk attire stroll through cobblestone streets under a sky filled with floating airships.",
            "An ancient, towering library with endless shelves filled with glowing books. Floating staircases and platforms hover in mid-air, while beams of warm golden light filter through stained-glass windows.",
            "A radiant warrior in gleaming silver armor standing on the edge of a cosmic battlefield. Stars and nebulae swirl behind them, and their sword glows with divine energy as meteors streak across the sky.",
            "A peaceful sanctuary where advanced technology blends seamlessly with nature. Transparent domes house lush green environments, and holographic guides hover over serene walking paths surrounded by exotic flora.",
            "A vibrant market filled with merchants selling magical items: glowing potions, enchanted crystals, and mysterious artifacts. Creatures of all shapes and sizes browse the colorful stalls under a sky filled with floating lanterns.",
            "A vast desert landscape with golden sand dunes under a blazing sun. In the distance, a shimmering mirage reveals a floating palace made of glass and jewels, reflecting every color of the rainbow.",
            "A dimly lit alley filled with neon graffiti and flickering holographic advertisements. The air is thick with fog, and a mysterious figure in a hooded coat leans against a futuristic motorcycle.",
            "A dreamy garden scene with oversized teapots, floating teacups, and talking animals gathered around a long whimsical table. Flowers bloom in vibrant colors, and glowing fireflies hover in the twilight.",
            "A circle of hooded figures performing a mystical ritual on a cliff during a total solar eclipse. The sky is painted with dramatic hues of orange, purple, and black, while energy crackles in the air.",
            "A team of astronauts in high-tech suits walking through a dense extraterrestrial jungle filled with towering, bioluminescent trees and strange glowing plants. Exotic creatures with multiple eyes and iridescent wings watch from the shadows.",
            "An ancient stone temple covered in moss and vines, nestled deep in a misty rainforest. Golden sunlight streams through the trees, illuminating carved symbols on weathered stone pillars.",
            "A majestic phoenix with fiery wings soaring above a tranquil mountain lake, casting shimmering reflections on the water. Sparks and embers trail behind it as the sky glows with hues of orange and gold.",
            "A private tropical paradise with white sandy beaches, crystal-clear turquoise waters, and an overwater villa with an infinity pool. Palm trees sway gently under a vibrant sunset.",
            "A lively 1950s-style carnival filled with colorful tents, a grand Ferris wheel, and joyful crowds. Cotton candy vendors and performers in retro outfits entertain visitors under a pastel sky."
        ]

        prompts = [f'Generate an image according to the following prompt:\n{prompt}' for prompt in sentences]
        inference_t2i(args, prompts, args.save_root)

    elif args.inference_mode == "I2T":
        # I2T test samples
        # caption_prompts = ["<|image|>\nPlease describe the given image in detail", "<|image|>\n请详细描述图像中的内容"]
        # caption_prompts = ["<|image|>\nWhat does the image imply?"]
        # caption_prompts = ["<|image|>\n第一阶段和第二阶段模型的全局学习率分别是多少？"]
        caption_prompts = ["<|image|>\nPlease describe the given image in detail"]
        caption_images = "image.png"
        inference_i2t(args, caption_prompts, caption_images)

    