document.addEventListener('DOMContentLoaded', function () {
    var form = document.getElementById('signupForm');
    var userMessage = document.getElementById('UserMessage');

    form.addEventListener('submit', function (event) {
        event.preventDefault(); // Prevent the form from submitting initially

        // Validate all inputs
        var inputs = form.querySelectorAll('input, select');
        var isValid = true;

        inputs.forEach(function (input) {
            if (input.hasAttribute('required') && !input.value.trim()) {
                isValid = false;
                input.classList.add('error');
            } else {
                input.classList.remove('error');
            }
        });

        // Validate email format
        var email = document.getElementById('email');
        var emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailPattern.test(email.value)) {
            isValid = false;
            alert('Please enter a valid email address.');
        }

        // Validate password strength
        var password = document.getElementById('password').value;
        var confirmPassword = document.getElementById('confirm_password').value;
        var passwordPattern = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/;
        
        if (!passwordPattern.test(password)) {
            isValid = false;
            alert('Password must be at least 8 characters long and include an uppercase letter, a lowercase letter, a number, and a special character.');
        }

        // Validate passwords match
        if (password !== confirmPassword) {
            isValid = false;
            alert('Passwords do not match!');
        }

        // Ensure password has no spaces
        if (password.includes(' ')) {
            isValid = false;
            alert('Password cannot contain spaces.');
        }

        // Validate Terms and Conditions checkbox
        var termsCheckbox = document.getElementById('terms');
        if (!termsCheckbox.checked) {
            isValid = false;
            alert('You must agree to the terms and conditions.');
        }

        if (isValid) {
            userMessage.style.display = 'block';
            userMessage.textContent = 'Thank you for signing up! Redirecting to the main page...';
            userMessage.style.fontWeight = 'bold';
            userMessage.style.fontSize = '18px';
            setTimeout(function () {
                window.location.href = 'main.html';
            }, 2000); // Redirect to main page after 2 seconds
        } else {
            alert('Please fill in all required fields correctly.');
        }
    });
});
