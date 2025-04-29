import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class EjemploLector {
	
	public static void main(String[] args){
		try{
			BufferedReader br=new BufferedReader(new FileReader("C:\\Users\\Sofia\\Desktop\\10\\POO\\P2\\Ejemplo.txt"));
			String linea=br.readLine(); //lee una linea y avanza el cursor a la siguiente linea
			System.out.println(linea);
			linea=br.readLine();
			System.out.println(linea);
			linea=br.readLine();
			System.out.println(linea);
			br.close();
		}catch(FileNotFoundException ex){
			System.out.println("No se encontró el archivo");
		}catch(IOException ex){
			System.out.println("Error al leer el archivo");
		}
	}

}
