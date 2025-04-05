// Tab switching functionality
const loginBtn = document.querySelector("#login");
const registerBtn = document.querySelector("#register");
const loginForm = document.querySelector(".login-form");
const registerForm = document.querySelector(".register-form");

// Switch to login form
loginBtn.addEventListener('click', () => {
    loginBtn.style.backgroundColor = "#282242";
    registerBtn.style.backgroundColor = "rgba(255,255,255,0.2)";

    loginForm.style.left = "50%";
    registerForm.style.left = "-50%";

    loginForm.style.opacity = 1;
    registerForm.style.opacity = 0;
});

// Switch to register form
registerBtn.addEventListener('click', () => {
    loginBtn.style.backgroundColor = "rgba(255,255,255,0.2)";
    registerBtn.style.backgroundColor = "#282242";

    loginForm.style.left = "150%";
    registerForm.style.left = "50%";

    loginForm.style.opacity = 0;
    registerForm.style.opacity = 1;
});

// Utility: Validate email format
function isValidEmail(email) {
    const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return regex.test(email);
}

// Utility: Check password strength
function isStrongPassword(password) {
    return password.length >= 6;
}

// Form validation on submit
document.querySelectorAll('form').forEach(form => {
    form.addEventListener('submit', function (e) {
        let isValid = true;
        const inputs = this.querySelectorAll('input[required]');

        // Reset all borders
        inputs.forEach(input => {
            input.style.borderColor = "";
        });

        // Check empty fields
        inputs.forEach(input => {
            if (!input.value.trim()) {
                input.style.borderColor = "red";
                isValid = false;
            }
        });

        // Extra checks for registration form
        if (this.classList.contains('register-form')) {
            const emailInput = this.querySelector('input[name="email"]');
            const passwordInput = this.querySelector('input[name="password"]');

            if (!isValidEmail(emailInput.value.trim())) {
                emailInput.style.borderColor = "red";
                isValid = false;
            }

            if (!isStrongPassword(passwordInput.value.trim())) {
                passwordInput.style.borderColor = "red";
                isValid = false;
            }
        }

        if (!isValid) {
            e.preventDefault();
        }
    });
});
