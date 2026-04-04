import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;

class ApiService {
  // Cambia esta URL por la de tu backend en Railway una vez deployado
  static const String baseUrl = 'https://christo-wrap-backend-production.up.railway.app';

  /// Envia la imagen al backend y devuelve la imagen resultado en bytes.
  /// [imageBytes]: bytes de la imagen original
  /// [material]: "tela" o "plastico"
  static Future<Uint8List> wrapObject({
    required Uint8List imageBytes,
    String material = 'tela',
  }) async {
    final uri = Uri.parse('$baseUrl/wrap?material=$material');

    final request = http.MultipartRequest('POST', uri);
    request.files.add(
      http.MultipartFile.fromBytes(
        'file',
        imageBytes,
        filename: 'photo.jpg',
      ),
    );

    final streamedResponse = await request.send().timeout(
      const Duration(minutes: 3),
    );

    if (streamedResponse.statusCode != 200) {
      final body = await streamedResponse.stream.bytesToString();
      throw Exception('Error del servidor: $body');
    }

    final responseBody = await streamedResponse.stream.bytesToString();
    final json = jsonDecode(responseBody) as Map<String, dynamic>;

    final resultB64 = json['result_image'] as String;
    return base64Decode(resultB64);
  }
}
