function toggleInfoPopup() {
    const popup = document.getElementById("infoPopup");
    if (popup.style.display === "none" || popup.style.display === "") {
        popup.style.display = "block";
    } else {
        popup.style.display = "none";
    }
}

function showImage(img, caption) {
    // Toon de modal
    const modal = document.getElementById("imageModal");
    modal.style.display = "block";

    // Stel de afbeelding en het bijschrift in
    const modalImg = document.getElementById("modalImage");
    const modalCaption = document.getElementById("modalCaption");

    modalImg.src = img.src;
    modalCaption.textContent = caption;
}

function closeImage() {
    // Sluit de modal
    const modal = document.getElementById("imageModal");
    modal.style.display = "none";
}


function toggleChunk(chunkId) {
    const chunkElement = document.getElementById(chunkId);
    const button = chunkElement.querySelector("button");

    if (chunkElement.classList.contains("expanded")) {
        chunkElement.classList.remove("expanded");
        button.textContent = "Lees meer";
    } else {
        chunkElement.classList.add("expanded");
        button.textContent = "Lees minder";
    }
}