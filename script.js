// i made a div with class name "circle" now i want use btn id to apear this circle with animations and circle should have 7 gradiant colors and it animated

const btn = document.getElementById("btn");
const circle = document.querySelector(".circle");

btn.addEventListener("click", () => {

    circle.classList.add("animate");
    circle.addEventListener("animationend", () => {        
        circle.classList.remove("animate");
        
    });
    // animation codes
    circle.style.background = "linear-gradient(45deg, red, orange, yellow, green, blue, indigo, violet)";
    circle.style.animation = "rotate 2s linear infinite";


});
