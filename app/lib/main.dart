import 'package:flutter/material.dart';
import 'screens/home_screen.dart';

void main() {
  runApp(const ChristoApp());
}

class ChristoApp extends StatelessWidget {
  const ChristoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Christo Wrap',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF8B6914),
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
        fontFamily: 'Georgia',
      ),
      home: const HomeScreen(),
    );
  }
}
