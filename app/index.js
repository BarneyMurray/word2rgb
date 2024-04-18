import { AutoModel, AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.0';


const loader = document.getElementById("loader")

// wait for tokenizer and model to load, then hide loader
const tokenizer = await AutoTokenizer.from_pretrained('BarneyMurray0/word2rgb');
const model = await AutoModel.from_pretrained('BarneyMurray0/word2rgb');

loader.style.display = 'none'

document.getElementById('predictButton').addEventListener('click', predictRGB);

function applySigmoid(logits) {
    return logits.map(value => 1 / (1 + Math.exp(-value)));
}

function scaleBy255(logits) {
    return logits.map(value => Math.round(value * 255));
}

function updateHexCode(rgb) {
    let hexCode = document.getElementById('hexCode');
    hexCode.textContent = `#${componentToHex(rgb[0])}${componentToHex(rgb[1])}${componentToHex(rgb[2])}`;
}

function componentToHex(c) {
    let hex = c.toString(16);
    return hex.length == 1 ? "0" + hex : hex;
}

export async function predictRGB() {
    let textInput = document.getElementById('textInput').value.trim();
    if (textInput === '') {
        alert('Please enter some text.');
        return;
    }

    let inputs = await tokenizer(textInput);
    let { logits } = await model(inputs);

    let rgb = scaleBy255(applySigmoid(logits.data));

    updateColorSquare(rgb);
    updateHexCode(rgb)
}

function updateColorSquare(rgb) {
    let colorSquare = document.getElementById('colorSquare');
    colorSquare.style.backgroundColor = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}
