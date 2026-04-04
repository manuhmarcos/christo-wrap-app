import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/api_service.dart';
import 'result_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ImagePicker _picker = ImagePicker();
  bool _isProcessing = false;
  String _statusMessage = '';
  String _selectedMaterial = 'tela';

  Future<void> _pickAndProcess(ImageSource source) async {
    final XFile? picked = await _picker.pickImage(
      source: source,
      imageQuality: 90,
      maxWidth: 1024,
      maxHeight: 1024,
    );
    if (picked == null) return;

    setState(() {
      _isProcessing = true;
      _statusMessage = 'Detectando objeto principal...';
    });

    try {
      final Uint8List imageBytes = await picked.readAsBytes();

      setState(() => _statusMessage = 'Generando envolvimiento fotorrealista...\n(esto puede tomar 30-60 segundos)');

      final Uint8List resultBytes = await ApiService.wrapObject(
        imageBytes: imageBytes,
        material: _selectedMaterial,
      );

      if (!mounted) return;
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => ResultScreen(
            originalBytes: imageBytes,
            resultBytes: resultBytes,
          ),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Error: ${e.toString()}'),
          backgroundColor: Colors.red[800],
          duration: const Duration(seconds: 5),
        ),
      );
    } finally {
      if (mounted) {
        setState(() => _isProcessing = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1A1209),
      body: SafeArea(
        child: _isProcessing ? _buildProcessing() : _buildHome(),
      ),
    );
  }

  Widget _buildProcessing() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const SizedBox(
            width: 80,
            height: 80,
            child: CircularProgressIndicator(
              strokeWidth: 6,
              color: Color(0xFFD4A843),
            ),
          ),
          const SizedBox(height: 32),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 32),
            child: Text(
              _statusMessage,
              textAlign: TextAlign.center,
              style: const TextStyle(
                color: Color(0xFFD4A843),
                fontSize: 16,
                height: 1.5,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHome() {
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const SizedBox(height: 16),
          // Header
          const Text(
            'CHRISTO\nWRAP',
            style: TextStyle(
              color: Color(0xFFD4A843),
              fontSize: 48,
              fontWeight: FontWeight.bold,
              height: 1.1,
              letterSpacing: 4,
            ),
          ),
          const SizedBox(height: 8),
          const Text(
            'Envuelve el mundo al estilo\nChristo y Jeanne-Claude',
            style: TextStyle(
              color: Color(0xFF9E8B6A),
              fontSize: 15,
              height: 1.4,
            ),
          ),
          const SizedBox(height: 40),
          // Material selector
          const Text(
            'MATERIAL',
            style: TextStyle(
              color: Color(0xFF9E8B6A),
              fontSize: 12,
              letterSpacing: 2,
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              _materialChip('tela', 'Tela', '🧵'),
              const SizedBox(width: 12),
              _materialChip('plastico', 'Plastico', '✨'),
            ],
          ),
          const Spacer(),
          // Main image preview area
          Container(
            width: double.infinity,
            height: 260,
            decoration: BoxDecoration(
              border: Border.all(color: const Color(0xFF4A3A1A), width: 1.5),
              borderRadius: BorderRadius.circular(16),
            ),
            child: const Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.photo_camera_outlined, size: 64, color: Color(0xFF4A3A1A)),
                SizedBox(height: 16),
                Text(
                  'Toma o selecciona una foto',
                  style: TextStyle(color: Color(0xFF6A5A3A), fontSize: 14),
                ),
              ],
            ),
          ),
          const SizedBox(height: 24),
          // Buttons
          Row(
            children: [
              Expanded(
                child: _ActionButton(
                  icon: Icons.camera_alt,
                  label: 'CAMARA',
                  onTap: () => _pickAndProcess(ImageSource.camera),
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: _ActionButton(
                  icon: Icons.photo_library,
                  label: 'GALERIA',
                  onTap: () => _pickAndProcess(ImageSource.gallery),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
        ],
      ),
    );
  }

  Widget _materialChip(String value, String label, String emoji) {
    final bool selected = _selectedMaterial == value;
    return GestureDetector(
      onTap: () => setState(() => _selectedMaterial = value),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
        decoration: BoxDecoration(
          color: selected ? const Color(0xFFD4A843) : Colors.transparent,
          border: Border.all(
            color: selected ? const Color(0xFFD4A843) : const Color(0xFF4A3A1A),
            width: 1.5,
          ),
          borderRadius: BorderRadius.circular(24),
        ),
        child: Text(
          '$emoji  $label',
          style: TextStyle(
            color: selected ? const Color(0xFF1A1209) : const Color(0xFF9E8B6A),
            fontWeight: selected ? FontWeight.bold : FontWeight.normal,
            fontSize: 14,
          ),
        ),
      ),
    );
  }
}

class _ActionButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  const _ActionButton({
    required this.icon,
    required this.label,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 18),
        decoration: BoxDecoration(
          color: const Color(0xFFD4A843),
          borderRadius: BorderRadius.circular(14),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, color: const Color(0xFF1A1209), size: 22),
            const SizedBox(width: 10),
            Text(
              label,
              style: const TextStyle(
                color: Color(0xFF1A1209),
                fontWeight: FontWeight.bold,
                fontSize: 14,
                letterSpacing: 1.5,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
