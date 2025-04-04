// Variable Declaration

const loginBtn = document.querySelector("#login");
const registerBtn = document.querySelector("#register");
const loginForm = document.querySelector(".login-form");
const registerForm = document.querySelector(".register-form");

// Login button function
loginBtn.addEventListener('click', () => {
    loginBtn.style.backgroundColor = "#282242";
    registerBtn.style.backgroundColor = "rgba(255,255,255,0.2)";

    loginForm.style.left = "50%";
    registerForm.style.left = "-50%";

    loginForm.style.opacity = 1;
    registerForm.style.opacity = 0;

})

registerBtn.addEventListener('click', () => {
    loginBtn.style.backgroundColor = "rgba(255,255,255,0.2)";
    registerBtn.style.backgroundColor = "#282242";

    loginForm.style.left = "150%";
    registerForm.style.left = "50%";

    loginForm.style.opacity = 0;
    registerForm.style.opacity = 1;

})

// Form Validation
document.querySelectorAll(".input-submit").forEach(button => {
    button.addEventListener("click", (e) => {
        e.preventDefault();

        const form = button.closest(".form-inputs");
        const inputs = form.querySelectorAll("input");

        let isValid = true;
        inputs.forEach(input => {
            if (input.value.trim() === "") {
                isValid = false;
                input.style.borderColor = "red";
            } else {
                input.style.borderColor = "#ccc";
            }

            // Email format check for register form
            if (input.type === "text" && input.placeholder === "Email") {
                const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
                if (!emailRegex.test(input.value)) {
                    isValid = false;
                    input.style.borderColor = "red";
                }
            }
        });

        if (isValid) {
            alert("Form submitted successfully!");
        } else {
            alert("Please fill in all fields correctly.");
        }
    });
});