// Facebook Cookie Exporter
// Run this in your browser console while logged into Facebook
//
// Instructions:
// 1. Go to facebook.com and make sure you're logged in
// 2. Press F12 to open Developer Tools
// 3. Click the "Console" tab
// 4. Paste this entire script and press Enter
// 5. Copy the output and paste it into your .env file

(function() {
    const cookieNames = ['datr', 'sb', 'c_user', 'xs', 'fr', 'locale'];
    const cookies = [];

    document.cookie.split(';').forEach(cookie => {
        const [name, value] = cookie.trim().split('=');
        if (cookieNames.includes(name)) {
            cookies.push({
                name: name,
                value: value,
                domain: '.facebook.com',
                path: '/'
            });
        }
    });

    const found = cookies.map(c => c.name);
    const missing = cookieNames.filter(name => !found.includes(name));

    if (missing.length > 0) {
        console.warn('Missing cookies:', missing.join(', '));
        console.warn('Make sure you are logged into Facebook and try again.');
    }

    if (cookies.length > 0) {
        const output = JSON.stringify(cookies);
        console.log('\n✅ Copy this line into your .env file:\n');
        console.log('FB_COOKIES=' + output);
        console.log('\n');

        // Also copy to clipboard if possible
        try {
            navigator.clipboard.writeText('FB_COOKIES=' + output);
            console.log('📋 Also copied to clipboard!');
        } catch (e) {
            // Clipboard access may be denied
        }
    } else {
        console.error('❌ No Facebook cookies found. Are you logged in?');
    }
})();
