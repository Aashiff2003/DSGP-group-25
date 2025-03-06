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

        // Validate passwords
        var password = document.getElementById('password').value;
        var confirmPassword = document.getElementById('confirm_password').value;

        if (password !== confirmPassword) {
            isValid = false;
            alert('Passwords do not match!');
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
            alert('Please fill in all required fields.');
        }
    });
});
